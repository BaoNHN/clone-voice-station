"""
voice/rvc_local.py
Local (this-machine) fallback for RVC training + conversion, used when the
Colab server is not configured or unreachable. Training runs as a subprocess
in an isolated venv; conversion runs in-process.
"""

import glob
import io
import os
import queue
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time

from database.database import VOICE_MODELS_DIR
from engine.server_log import get_logger

logger = get_logger()

BASE_DIR       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOCAL_RVC_DIR  = os.path.join(BASE_DIR, "local_rvc")  # cloned RVC-WebUI repo + venv + pretrained assets (gitignored)
RVC_REPO_DIR   = os.path.join(LOCAL_RVC_DIR, "RVC")
RVC_REPO_URL   = "https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI"
RVC_VENV_DIR   = os.path.join(LOCAL_RVC_DIR, "venv")

# Bundled ffmpeg on PATH, needed by _slice_and_normalize()'s pydub calls
# (works around a conda-forge ffmpeg launch failure on this machine).
_BUNDLED_FFMPEG_DIR = os.path.join(BASE_DIR, "bin")
if os.path.isfile(os.path.join(_BUNDLED_FFMPEG_DIR, "ffmpeg.exe")):
    os.environ["PATH"] = _BUNDLED_FFMPEG_DIR + os.pathsep + os.environ.get("PATH", "")

_HF_BASE = "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main"
# Pretrained assets downloaded once on first run; see ensure_set_up().
ASSETS = {
    os.path.join(RVC_REPO_DIR, "assets", "pretrained_v2", "f0G40k.pth"):        f"{_HF_BASE}/pretrained_v2/f0G40k.pth",
    os.path.join(RVC_REPO_DIR, "assets", "pretrained_v2", "f0D40k.pth"):        f"{_HF_BASE}/pretrained_v2/f0D40k.pth",
    os.path.join(RVC_REPO_DIR, "assets", "hubert_base", "config.json"):             f"{_HF_BASE}/hubert_base/config.json",
    os.path.join(RVC_REPO_DIR, "assets", "hubert_base", "preprocessor_config.json"): f"{_HF_BASE}/hubert_base/preprocessor_config.json",
    os.path.join(RVC_REPO_DIR, "assets", "hubert_base", "pytorch_model.bin"):        f"{_HF_BASE}/hubert_base/pytorch_model.bin",
    os.path.join(RVC_REPO_DIR, "assets", "rmvpe", "rmvpe.pt"):                  f"{_HF_BASE}/rmvpe.pt",
}

SAMPLE_RATE        = 40000   # RVC v2 standard, matches colab/voice_server.ipynb
F0_METHOD           = "rmvpe"
RVC_VERSION         = "v2"
PITCH_DEFAULT       = 0
INDEX_RATE_DEFAULT  = 0.75
PROTECT             = 0.33
SAVE_EVERY          = 5      # small enough that a checkpoint exists by the time early stopping can trigger
BATCH_SIZE_GPU_HIGH_VRAM = 8   # matches colab/voice_server.ipynb, tuned for a 16GB T4
BATCH_SIZE_GPU_LOW_VRAM  = 4   # cards under LOW_VRAM_THRESHOLD_GB
LOW_VRAM_THRESHOLD_GB    = 10  # same cutoff colab/voice_server.ipynb's GPU-check cell uses
BATCH_SIZE_CPU      = 4
EPOCHS_GPU          = 200    # matches colab/voice_server.ipynb TOTAL_EPOCHS
EPOCHS_CPU          = 40     # bounded fallback; CPU has no realistic path to 200 epochs

# Early-stopping thresholds for the CPU fallback path (GPU/Colab rarely needs
# cutting short).
EARLY_STOP_MIN_EPOCHS = 15
EARLY_STOP_PATIENCE    = 5
EARLY_STOP_MIN_DELTA   = 0.01

# Kills a training subprocess that produces no output at all for this long
# (treated as hung, not slow).
STALL_TIMEOUT_SEC = 600

# Bounds the venv dependency install in ensure_set_up(); a hung pip has no
# other timeout.
PIP_INSTALL_TIMEOUT_SEC = 900

# Matches train/train.py's own per-step log line.
_LOSS_LINE_RE = re.compile(
    r"loss_gen=([\d.]+),\s*loss_fm=([\d.]+),\s*loss_mel=([\d.]+),\s*loss_kl=([\d.]+)"
)
_EPOCH_MARKER = "====> "  # train/train.py's per-epoch marker, language-independent

# Bounds a single convert_local() call (covers a cold VC() load).
CONVERT_TIMEOUT_SEC = 150

# Serializes cached-VC() access across concurrent /api/speak calls (single
# local GPU, no real parallelism to offer).
_convert_lock = threading.Lock()

_rvc_import_ready = False        # set once, lazily, by _prepare_rvc_import()
_convert_vc = None                # cached in-process VC() instance
_convert_loaded_speaker = None    # which speaker's weights _convert_vc currently has loaded


def device() -> str:
    """"cuda:0" if this machine has a usable GPU, else "cpu"."""
    import torch
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def is_gpu() -> bool:
    return device() != "cpu"


def _epochs_for_device() -> int:
    return EPOCHS_GPU if is_gpu() else EPOCHS_CPU


def _batch_size_for_device() -> int:
    if not is_gpu():
        return BATCH_SIZE_CPU
    import torch
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    return BATCH_SIZE_GPU_HIGH_VRAM if vram_gb >= LOW_VRAM_THRESHOLD_GB else BATCH_SIZE_GPU_LOW_VRAM


def is_set_up() -> bool:
    """True if the RVC repo, its isolated training venv, and the pretrained
    assets are already present locally (no network access needed)."""
    return (os.path.isdir(RVC_REPO_DIR) and os.path.exists(_venv_python())
            and all(os.path.exists(p) for p in ASSETS))


def _venv_python() -> str:
    if os.name == "nt":
        return os.path.join(RVC_VENV_DIR, "Scripts", "python.exe")
    return os.path.join(RVC_VENV_DIR, "bin", "python")


# Minimal set of packages the training scripts actually need beyond what
# rag_env's own torch/numpy/scipy/librosa/soundfile/faiss already provide.
TRAINING_PACKAGES = [
    "av", "ffmpeg-python", "matplotlib", "praat-parselmouth",
    "scikit-learn", "tensorboard", "transformers",
]


SETUP_LOCK_TIMEOUT_SEC = 1800  # generous -- a legitimate first-run setup can itself take a while


def _acquire_setup_lock(report) -> int:
    """Cross-process mutex around ensure_set_up()'s repo-clone/venv-build/
    asset-download sequence, so two concurrent setup attempts can't corrupt
    the shared local_rvc/ directory. Returns an fd to release via
    _release_setup_lock(). A stale lock is reclaimed after
    SETUP_LOCK_TIMEOUT_SEC rather than blocking forever."""
    os.makedirs(LOCAL_RVC_DIR, exist_ok=True)
    lock_path = os.path.join(LOCAL_RVC_DIR, ".setup.lock")
    waited = 0
    warned = False
    while True:
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_RDWR)
            os.write(fd, str(os.getpid()).encode())
            return fd
        except FileExistsError:
            try:
                age = time.time() - os.path.getmtime(lock_path)
            except OSError:
                age = SETUP_LOCK_TIMEOUT_SEC + 1
            if age > SETUP_LOCK_TIMEOUT_SEC:
                try:
                    os.remove(lock_path)
                except OSError:
                    pass
                continue
            if not warned:
                report("Another local RVC setup/training run is already in progress on this "
                       "machine -- waiting for it to finish before starting.")
                warned = True
            if waited >= SETUP_LOCK_TIMEOUT_SEC:
                raise RuntimeError(
                    f"Timed out after {SETUP_LOCK_TIMEOUT_SEC // 60} min waiting for another "
                    f"local RVC setup/training run to release its lock."
                )
            time.sleep(2)
            waited += 2


def _release_setup_lock(fd: int):
    try:
        os.close(fd)
    except OSError:
        pass
    try:
        os.remove(os.path.join(LOCAL_RVC_DIR, ".setup.lock"))
    except OSError:
        pass


def ensure_set_up(progress_cb=None):
    """Clones RVC-Project's WebUI repo, builds an isolated venv for training,
    and downloads the pretrained assets. Only does what's missing. Raises on
    failure -- callers should treat that as "local fallback unavailable".
    Training needs its own venv (dependency conflicts with this app's own
    FastAPI stack); conversion runs directly in this process instead."""
    import urllib.request

    def report(msg):
        logger.info(f"[RVC-local] {msg}")
        if progress_cb:
            progress_cb(msg)

    lock_fd = _acquire_setup_lock(report)
    try:
        _ensure_set_up_locked(report)
    finally:
        _release_setup_lock(lock_fd)


def _ensure_set_up_locked(report):
    import urllib.request

    os.makedirs(LOCAL_RVC_DIR, exist_ok=True)
    if not os.path.isdir(RVC_REPO_DIR):
        report("Cloning RVC-Project/Retrieval-based-Voice-Conversion-WebUI (first run only)…")
        subprocess.run(["git", "clone", "--depth=1", RVC_REPO_URL, RVC_REPO_DIR],
                       check=True, timeout=PIP_INSTALL_TIMEOUT_SEC)

    # Marker lives in the venv dir so a previous attempt that cloned but died
    # mid pip-install doesn't get skipped just because the repo folder exists.
    deps_marker = os.path.join(RVC_VENV_DIR, ".deps_installed")
    if not os.path.exists(deps_marker):
        if os.path.isdir(RVC_VENV_DIR):
            shutil.rmtree(RVC_VENV_DIR, ignore_errors=True)
        report("Creating isolated venv for RVC training (first run only)…")
        # --system-site-packages: reuses this process's own torch instead of a
        # second ~2-3GB download; packages installed into the venv below still
        # take priority for anything run inside it.
        subprocess.run([sys.executable, "-m", "venv", "--system-site-packages", RVC_VENV_DIR],
                       check=True, timeout=120)

        report(f"Installing training dependencies into isolated venv "
               f"({', '.join(TRAINING_PACKAGES)})…")
        try:
            subprocess.run(
                [_venv_python(), "-m", "pip", "install", "-q",
                 "--index-url", "https://pypi.org/simple",
                 "--timeout", "120", "--retries", "5",
                 "--no-cache-dir",
                 *TRAINING_PACKAGES],
                check=True, timeout=PIP_INSTALL_TIMEOUT_SEC,
            )
        except subprocess.TimeoutExpired:
            raise RuntimeError(
                f"pip install stalled for over {PIP_INSTALL_TIMEOUT_SEC // 60} minutes with no "
                f"progress (not a normal slow-but-working download) and was killed."
            )
        open(deps_marker, "w").close()
        report("Training dependencies installed.")

    # Needed in this process's own site-packages, not just the venv's --
    # convert_local() (in-process) also needs faiss for the index blend.
    try:
        import faiss  # noqa: F401
    except ImportError:
        report("Installing faiss-cpu (needed for conversion, not just training)…")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "faiss-cpu>=1.7.4"],
                       check=True, timeout=PIP_INSTALL_TIMEOUT_SEC)

    for dest, url in ASSETS.items():
        if os.path.exists(dest):
            continue
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        report(f"Downloading pretrained asset: {os.path.basename(dest)} (first run only)…")
        urllib.request.urlretrieve(url, dest)


def _model_paths(speaker_id: str) -> tuple[str, str]:
    """Same voice_storage/<speaker_id>/ layout engine/voice_engine.py uses
    for Colab-trained models, so a local model is indistinguishable to
    every other caller."""
    model_dir = os.path.join(VOICE_MODELS_DIR, speaker_id)
    return os.path.join(model_dir, f"{speaker_id}.pth"), os.path.join(model_dir, f"{speaker_id}.index")


def has_local_model(speaker_id: str) -> bool:
    pth_path, index_path = _model_paths(speaker_id)
    return os.path.exists(pth_path) and os.path.exists(index_path)


def _prepare_rvc_import():
    """Makes the cloned RVC repo importable from this process and chdir()s
    into it, once, lazily. Idempotent. Reimplements the small pieces of
    infer/cli.py needed instead of importing it directly, and imports
    sklearn/pandas/faiss/torch in a fixed order first (avoids a native-
    extension import-order crash on Windows)."""
    global _rvc_import_ready
    if _rvc_import_ready:
        return

    # Fixed import order avoids a native-extension crash on Windows.
    import sklearn  # noqa: F401
    import pandas  # noqa: F401
    import faiss  # noqa: F401
    import torch  # noqa: F401

    if RVC_REPO_DIR not in sys.path:
        sys.path.insert(0, RVC_REPO_DIR)
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("index_root", os.path.join(RVC_REPO_DIR, "logs"))
    os.environ.setdefault("outside_index_root", os.path.join(RVC_REPO_DIR, "assets", "indices"))
    os.environ.setdefault("rmvpe_root", os.path.join(RVC_REPO_DIR, "assets", "rmvpe"))
    os.chdir(RVC_REPO_DIR)
    _rvc_import_ready = True


def _create_rvc_config():
    """Builds a configs.config.Config() without letting its internal
    argparse.parse_args() see this process's own sys.argv."""
    from configs.config import Config

    original_argv = sys.argv[:]
    sys.argv = [sys.argv[0]]
    try:
        return Config()
    finally:
        sys.argv = original_argv


def _get_convert_vc():
    """Returns the cached in-process VC() instance, building it on first use.
    Caller must already hold _convert_lock."""
    global _convert_vc
    if _convert_vc is None:
        _prepare_rvc_import()
        from infer.vc.modules import VC

        config = _create_rvc_config()
        _convert_vc = VC(config)
        logger.info(f"[RVC-local] In-process VC initialized, device={config.device}")
    return _convert_vc


def _convert_with_timeout(pth_path: str, index_path: str, in_path: str,
                           pitch: int, index_rate: float):
    """Runs the cached VC.vc_single() call on a background thread with a
    wait bound, so a hang can't block the caller forever. On timeout the
    background thread (and its lock) keep running until it actually
    finishes."""
    result_box = queue.Queue(maxsize=1)

    def _worker():
        global _convert_loaded_speaker
        try:
            with _convert_lock:
                vc = _get_convert_vc()
                if _convert_loaded_speaker != os.path.basename(pth_path):
                    os.environ["weight_root"] = os.path.dirname(pth_path)
                    vc.get_vc(os.path.basename(pth_path))
                    _convert_loaded_speaker = os.path.basename(pth_path)
                status, result = vc.vc_single(
                    0, in_path, pitch, F0_METHOD, index_path, index_rate, 0, 1.0, PROTECT,
                )
            result_box.put(("ok", status, result))
        except Exception as e:
            result_box.put(("error", e, None))

    threading.Thread(target=_worker, daemon=True).start()
    try:
        outcome, status_or_error, result = result_box.get(timeout=CONVERT_TIMEOUT_SEC)
    except queue.Empty:
        return "timeout", None, None
    return outcome, status_or_error, result


def convert_local(audio_bytes: bytes, speaker_id: str, pitch: int = None,
                   index_rate: float = None, mime: str = "audio/mp3") -> bytes | None:
    """Runs voice conversion using the speaker's locally-available model
    (downloaded from Colab, or trained locally by train_speaker_local()).
    Keeps one VC() instance cached in-process for its whole lifetime (see
    _get_convert_vc()); only per-speaker weights reload on a speaker change.
    Returns None if no local model is available or on any failure --
    callers should fall back to unconverted TTS audio."""
    if not has_local_model(speaker_id):
        return None
    try:
        ensure_set_up()
    except Exception as e:
        logger.warning(f"[RVC-local] Local setup unavailable, can't convert for {speaker_id}: {e}")
        return None

    pth_path, index_path = _model_paths(speaker_id)
    pitch      = PITCH_DEFAULT if pitch is None else pitch
    index_rate = INDEX_RATE_DEFAULT if index_rate is None else index_rate
    in_suffix  = ".wav" if "wav" in mime else ".mp3"

    with tempfile.NamedTemporaryFile(suffix=in_suffix, delete=False) as fin:
        fin.write(audio_bytes)
        in_path = fin.name

    try:
        outcome, status_or_error, result = _convert_with_timeout(
            pth_path, index_path, in_path, pitch, index_rate,
        )
        if outcome == "timeout":
            logger.error(f"[RVC-local] convert timed out for {speaker_id} after {CONVERT_TIMEOUT_SEC}s")
            return None
        if outcome == "error":
            logger.error(f"[RVC-local] convert failed for {speaker_id} on {device()}: {status_or_error}")
            return None
        if not result or result[0] is None or result[1] is None:
            logger.error(f"[RVC-local] convert failed for {speaker_id} on {device()}: {status_or_error}")
            return None

        import soundfile as sf

        buf = io.BytesIO()
        sf.write(buf, result[1], result[0], format="WAV")
        return buf.getvalue()
    except Exception as e:
        logger.error(f"[RVC-local] convert failed for {speaker_id} on {device()}: {e}")
        return None
    finally:
        if os.path.exists(in_path):
            os.unlink(in_path)


def _slice_and_normalize(src: str, out_dir: str, sr: int, min_ms: int = 3000, max_ms: int = 8000) -> int:
    from pydub import AudioSegment
    from pydub.silence import split_on_silence

    audio = AudioSegment.from_file(src)
    audio = audio.set_frame_rate(sr).set_channels(1)
    audio = audio.apply_gain(-20.0 - audio.dBFS)

    chunks = split_on_silence(audio, min_silence_len=300, silence_thresh=audio.dBFS - 16, keep_silence=150)

    stem = os.path.splitext(os.path.basename(src))[0]
    saved, buf = 0, AudioSegment.empty()
    for chunk in chunks:
        buf += chunk
        while len(buf) >= min_ms:
            seg = buf[:max_ms]
            buf = buf[max_ms:]
            seg.export(os.path.join(out_dir, f"{stem}_{saved:04d}.wav"), format="wav")
            saved += 1
    if len(buf) >= min_ms:
        buf.export(os.path.join(out_dir, f"{stem}_{saved:04d}.wav"), format="wav")
        saved += 1
    return saved


def _run_step(cmd, label, cwd):
    """Runs a training subprocess step; raises with both stdout and stderr
    tails, since some of these scripts log failures to stdout only."""
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if result.returncode != 0:
        detail = (result.stdout[-1000:] + "\n" + result.stderr[-1000:]).strip()
        raise RuntimeError(f"{label} failed (exit {result.returncode}):\n{detail}")
    return result.stdout


def _write_train_config(exp_dir: str):
    """Writes config.json into the experiment dir, which train.utils.get_hparams()
    requires with no fallback. v1/40k.json is the correct template for
    SAMPLE_RATE=40k regardless of RVC_VERSION."""
    src = os.path.join(RVC_REPO_DIR, "configs", "v1", "40k.json")
    shutil.copy2(src, os.path.join(exp_dir, "config.json"))


def _run_training_with_early_stop(venv_python: str, exp_dir: str, speaker_id: str,
                                    batch_size: int, epochs: int, report) -> int:
    """Runs train/train.py, watching its live output so training can stop
    early once the generator loss hasn't improved for EARLY_STOP_PATIENCE
    fresh readings, instead of always running the full epoch count. Returns
    the last epoch reached (< epochs if stopped early). Raises RuntimeError
    on a subprocess failure that isn't our own early-stop termination."""
    # -c 1 caches the whole dataset in VRAM once (small personal-voice
    # datasets fit even on 4GB cards); left at 0 on CPU where there's no
    # transfer to cache.
    cache_in_gpu = "1" if is_gpu() else "0"
    proc = subprocess.Popen([
        venv_python, "-m", "train.train",
        "-e", speaker_id, "-sr", "40k", "-f0", "1", "-bs", str(batch_size),
        "-g", "0", "-te", str(epochs), "-se", str(SAVE_EVERY),
        "-pg", os.path.join("assets", "pretrained_v2", "f0G40k.pth"),
        "-pd", os.path.join("assets", "pretrained_v2", "f0D40k.pth"),
        "-l", "1", "-c", cache_in_gpu, "-sw", "0", "-v", RVC_VERSION,
    ], cwd=RVC_REPO_DIR, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)

    # Reading proc.stdout directly blocks with no timeout; pump it on a
    # background thread through a queue so STALL_TIMEOUT_SEC can apply.
    lines = queue.Queue()

    def _pump():
        try:
            for line in proc.stdout:
                lines.put(line)
        finally:
            lines.put(None)  # sentinel: stdout closed (process exited)

    threading.Thread(target=_pump, daemon=True).start()

    epoch           = 0
    last_metric     = None
    last_epoch      = 0       # epoch of the most recent fresh reading, improved or not
    metric_is_fresh = False  # a NEW loss line arrived since the last epoch boundary
    best_metric     = None
    best_epoch      = 0
    no_improve      = 0
    stopped_early   = False
    stalled         = False

    try:
        while True:
            try:
                line = lines.get(timeout=STALL_TIMEOUT_SEC)
            except queue.Empty:
                report(f"No output for {STALL_TIMEOUT_SEC // 60} minutes -- assuming the "
                       f"training process is stuck and stopping it.")
                stalled = True
                proc.terminate()
                break

            if line is None:  # stdout closed -- process has exited
                break

            m = _LOSS_LINE_RE.search(line)
            if m:
                last_metric = sum(float(g) for g in m.groups())
                metric_is_fresh = True
                continue
            if _EPOCH_MARKER not in line:
                continue

            epoch += 1
            # Only a fresh loss reading counts toward patience (evaluating
            # every epoch double-counted stale readings).
            if metric_is_fresh:
                last_epoch = epoch
                if epoch < EARLY_STOP_MIN_EPOCHS:
                    report(f"Epoch {epoch}/{epochs} — loss {last_metric:.3f} "
                           f"(before epoch {EARLY_STOP_MIN_EPOCHS} -- not yet counted toward early stop)")
                else:
                    if best_metric is None or last_metric < best_metric - EARLY_STOP_MIN_DELTA:
                        best_metric, best_epoch, no_improve = last_metric, epoch, 0
                    else:
                        no_improve += 1
                    report(f"Epoch {epoch}/{epochs} — loss {last_metric:.3f} "
                           f"(best {best_metric:.3f} @ epoch {best_epoch}, "
                           f"{no_improve}/{EARLY_STOP_PATIENCE} without improvement)")
                metric_is_fresh = False
            elif last_metric is not None:
                report(f"Epoch {epoch}/{epochs} — no new loss reading yet "
                       f"(last: {last_metric:.3f} @ epoch {last_epoch})")

            if epoch >= EARLY_STOP_MIN_EPOCHS and no_improve >= EARLY_STOP_PATIENCE:
                report(f"No improvement for {EARLY_STOP_PATIENCE} epochs -- stopping early "
                       f"at epoch {epoch} (target was {epochs}) to save time.")
                stopped_early = True
                proc.terminate()
                break
    finally:
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()

    if stalled:
        raise RuntimeError(
            f"Training stalled at epoch {epoch} -- no output for {STALL_TIMEOUT_SEC // 60} "
            f"minutes, process was terminated. This usually means a hang (e.g. a stuck "
            f"subprocess/DataLoader worker), not a normal crash -- check for OS/driver-level "
            f"issues rather than a Python traceback."
        )
    if not stopped_early and proc.returncode != 0:
        raise RuntimeError(f"Training failed (exit code {proc.returncode}) -- check server logs above.")

    return epoch


def _finalize_from_checkpoint(venv_python: str, exp_dir: str, speaker_id: str, epoch_reached: int):
    """Builds the final .pth from the latest raw checkpoint when early
    stopping cut the run short (train.py's own save step only fires at the
    full epoch target). Runs inside the training venv."""
    g_checkpoints = sorted(glob.glob(os.path.join(exp_dir, "G_*.pth")),
                            key=os.path.getmtime, reverse=True)
    if not g_checkpoints:
        raise RuntimeError("Early-stopped before any checkpoint was saved -- nothing to finalize.")

    config_path = os.path.join(exp_dir, "config.json")
    snippet = f"""
import json, torch
from train.utils import HParams
from train.process_ckpt import savee

hps = HParams(**json.load(open(r"{config_path}", encoding="utf-8")))
ckpt = torch.load(r"{g_checkpoints[0]}", map_location="cpu", weights_only=False)["model"]
result = savee(ckpt, {SAMPLE_RATE}, 1, "{speaker_id}", {epoch_reached}, "{RVC_VERSION}", hps)
print("SAVEE_RESULT:", result)
"""
    result = subprocess.run([venv_python, "-c", snippet], cwd=RVC_REPO_DIR,
                             capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Finalizing early-stopped model crashed:\n{result.stderr[-1500:]}")
    if "Traceback" in result.stdout:
        raise RuntimeError(f"savee() failed while finalizing early-stopped model:\n{result.stdout[-1500:]}")


def train_speaker_local(speaker_id: str, sample_files: list, progress_cb=None) -> tuple[str, str]:
    """Full local training pipeline for one speaker: slice/normalize ->
    preprocess -> F0 -> HuBERT features -> train RVC v2 -> FAISS index ->
    save into voice_storage/<speaker_id>/. Runs the same scripts webui.py
    itself shells out to, inside the isolated venv from ensure_set_up().
    sample_files: list of (filename, bytes) tuples. Raises RuntimeError (or
    lets a subprocess error propagate) on failure -- the caller turns that
    into a "failed" profile status."""
    def report(msg):
        logger.info(f"[RVC-local][{speaker_id}] {msg}")
        if progress_cb:
            progress_cb(msg)

    ensure_set_up(progress_cb=progress_cb)
    venv_python = _venv_python()

    dev        = device()
    epochs     = _epochs_for_device()
    batch_size = _batch_size_for_device()
    report(f"Training on {dev} ({epochs} epochs, batch size {batch_size})"
           + ("" if is_gpu() else " -- CPU training is slow, this may take a long while."))

    raw_dir    = tempfile.mkdtemp(prefix=f"raw_{speaker_id}_")
    sliced_dir = os.path.join(LOCAL_RVC_DIR, f"dataset_{speaker_id}_sliced")
    try:
        for fname, data in sample_files:
            with open(os.path.join(raw_dir, fname), "wb") as f:
                f.write(data)

        os.makedirs(sliced_dir, exist_ok=True)
        exp_dir = os.path.join(RVC_REPO_DIR, "logs", speaker_id)  # full path -- preprocess/extract_* want this
        os.makedirs(exp_dir, exist_ok=True)
        os.makedirs(os.path.join(RVC_REPO_DIR, "assets", "weights"), exist_ok=True)

        report("Slicing & normalizing samples…")
        sources = (glob.glob(os.path.join(raw_dir, "*.wav")) +
                   glob.glob(os.path.join(raw_dir, "*.mp3")) +
                   glob.glob(os.path.join(raw_dir, "*.webm")) +
                   glob.glob(os.path.join(raw_dir, "*.ogg")) +
                   glob.glob(os.path.join(raw_dir, "*.m4a")))
        if not sources:
            raise RuntimeError("No audio files found among the uploaded samples.")
        total_segs = sum(_slice_and_normalize(f, sliced_dir, SAMPLE_RATE) for f in sources)
        report(f"{total_segs} segments produced.")
        if total_segs < 20:
            report("Few segments -- quality may be lower than ideal, continuing anyway.")

        report("Preprocessing…")
        n_p = min(4, os.cpu_count() or 4)
        # Must run as "-m train.preprocess", not a bare file path (a bare
        # path causes a circular import against train.py).
        _run_step([venv_python, "-m", "train.preprocess",
                   sliced_dir, str(SAMPLE_RATE), str(n_p), exp_dir, "False", "3.7"],
                  "Preprocessing", cwd=RVC_REPO_DIR)

        report("Extracting F0 (RMVPE)…")
        _run_step([venv_python, "-m", "train.dataset.extract_f0",
                   "cpu", exp_dir, str(n_p), F0_METHOD],
                  "F0 extraction", cwd=RVC_REPO_DIR)

        report(f"Extracting HuBERT features on {dev}…")
        _run_step([venv_python, "-m", "train.dataset.extract_hubert_feature",
                   dev, "1", "0", exp_dir, RVC_VERSION, "False"],
                  "Feature extraction", cwd=RVC_REPO_DIR)

        report("Building training filelist…")
        gt_wavs_dir = os.path.join(exp_dir, "0_gt_wavs")
        feature_dir = os.path.join(exp_dir, "3_feature768")
        f0_dir      = os.path.join(exp_dir, "2a_f0")
        f0nsf_dir   = os.path.join(exp_dir, "2b-f0nsf")

        wav_map  = {os.path.splitext(os.path.basename(w))[0]: w
                    for w in sorted(glob.glob(os.path.join(gt_wavs_dir, "*.wav")))}
        feat_map = {os.path.splitext(os.path.basename(f))[0]: f
                    for f in sorted(glob.glob(os.path.join(feature_dir, "*.npy")))}
        common = sorted(set(wav_map) & set(feat_map))
        if not common:
            raise RuntimeError("No matching wav/feature pairs -- preprocessing or feature extraction failed.")

        lines = [f"{wav_map[s]}|{feat_map[s]}|{os.path.join(f0_dir, s + '.wav.npy')}|"
                 f"{os.path.join(f0nsf_dir, s + '.wav.npy')}|0" for s in common]
        filelist_path = os.path.join(exp_dir, "filelist.txt")
        with open(filelist_path, "w") as fh:
            fh.write("\n".join(lines))
        report(f"Filelist: {len(lines)} entries.")

        _write_train_config(exp_dir)

        report(f"Training up to {epochs} epochs on {dev} "
               f"(early stop after {EARLY_STOP_PATIENCE} epochs without improvement, "
               f"no earlier than epoch {EARLY_STOP_MIN_EPOCHS})…")
        reached_epoch = _run_training_with_early_stop(
            venv_python, exp_dir, speaker_id, batch_size, epochs, report
        )
        report(f"Training stopped at epoch {reached_epoch} (target was {epochs}).")

        report("Building FAISS index…")
        outside_index_root = os.path.join(RVC_REPO_DIR, "assets", "_local_fallback_indices")
        n_cpu = min(4, os.cpu_count() or 4)
        _run_step([venv_python, "-m", "train.train_index",
                   speaker_id, RVC_VERSION, outside_index_root, str(n_cpu), "single"],
                  "Index training", cwd=RVC_REPO_DIR)

        # train.py only writes assets/weights/<name>.pth at the full epoch
        # target; an early-stopped run needs it built explicitly.
        trained_pth = os.path.join(RVC_REPO_DIR, "assets", "weights", f"{speaker_id}.pth")
        if not os.path.exists(trained_pth):
            report("Building final model from the last checkpoint (early-stopped run)…")
            _finalize_from_checkpoint(venv_python, exp_dir, speaker_id, reached_epoch)
        if not os.path.exists(trained_pth):
            raise RuntimeError(f"Training finished but no checkpoint found at {trained_pth}.")

        added_indexes = sorted(
            glob.glob(os.path.join(exp_dir, "added_IVF*.index")), key=os.path.getmtime, reverse=True
        )
        if not added_indexes:
            raise RuntimeError("Index training finished but no added_IVF*.index file was produced.")
        trained_index = added_indexes[0]

        out_pth, out_index = _model_paths(speaker_id)
        os.makedirs(os.path.dirname(out_pth), exist_ok=True)
        shutil.copy2(trained_pth, out_pth)
        shutil.copy2(trained_index, out_index)
        report(f"Exported model → {out_pth}")

        return out_pth, out_index
    finally:
        shutil.rmtree(raw_dir, ignore_errors=True)
        shutil.rmtree(sliced_dir, ignore_errors=True)
