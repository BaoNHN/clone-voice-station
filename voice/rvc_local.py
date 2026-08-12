"""
voice/rvc_local.py
Local (this-machine) fallback for RVC training + conversion, used by
voice/rvc_client.py when the Colab server (colab/voice_server.ipynb) is not
configured or unreachable -- voice cloning needs to keep working on a
deployment with no Colab session running. Both training and conversion run
the same modern, fairseq-free pipeline from a locally-cloned copy of
RVC-Project's WebUI repo (train/*.py for training, infer/cli.py for
conversion), invoked inside an isolated venv -- see ensure_set_up() for why
the venv, and convert_local()'s docstring for why not the rvc-python PyPI
package (its fairseq==0.12.2 dependency cannot even be imported on Python
3.12). Device is picked at runtime instead of hardcoded, unlike the Colab
notebook's own "cuda:0".

Device selection mirrors voice/stt.py's Whisper fallback: torch.cuda.is_available()
decides GPU vs CPU with no manual configuration needed. CPU training is
*technically* wired through (the RVC scripts accept a "cpu" device string same
as "cuda:0") but is realistically only a bounded-effort fallback, not a Colab
replacement -- a run that takes ~30-60 min on a T4 GPU can take many hours on
CPU, so _epochs_for_device() trains far fewer epochs on CPU to keep it
finite. Expect lower quality from a CPU-trained voice than from Colab/GPU.

Everything heavy (torch, faiss, pydub) is imported lazily inside functions,
same pattern as voice/stt.py's Whisper load, so a deployment that always has
Colab available never pays the import/clone/download cost.
"""

import glob
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

# Same bundled-ffmpeg PATH fix voice/stt.py applies for itself (rag_env's conda-forge
# ffmpeg fails to launch on this machine -- STATUS_ENTRYPOINT_NOT_FOUND, a DLL conflict
# with its dynamically-linked build) -- this module needs it too, for _slice_and_normalize()'s
# pydub calls, but doesn't import stt.py so never picked it up. Hit this for real: pydub's
# ffprobe-based AudioSegment.from_file() silently got no output from a broken/missing
# ffprobe and choked trying to json.loads() the empty result ("Expecting value: line 1
# column 1 (char 0)").
_BUNDLED_FFMPEG_DIR = os.path.join(BASE_DIR, "bin")
if os.path.isfile(os.path.join(_BUNDLED_FFMPEG_DIR, "ffmpeg.exe")):
    os.environ["PATH"] = _BUNDLED_FFMPEG_DIR + os.pathsep + os.environ.get("PATH", "")

_HF_BASE = "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main"
# colab/voice_server.ipynb (cell E) downloads a single hubert_base.pt (fairseq-format
# checkpoint) -- this repo's current HEAD no longer uses that at all. infer/hubert.py
# now loads a HuggingFace Transformers-format model directory instead
# (HubertModelWithFinalProj.from_pretrained(".../assets/hubert_base", local_files_only=True),
# requiring config.json + preprocessor_config.json + pytorch_model.bin there), confirmed
# against this repo's own docs/en/README.en.md download instructions:
#   hf download lj1995/VoiceConversionWebUI --include "hubert_base/*" --local-dir assets
# ~400MB total across all assets below.
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
# Small enough that a checkpoint always exists by the time early stopping (below)
# could plausibly trigger -- with the old value (20) an early stop before epoch 20
# would have had no checkpoint to build a final model from at all.
SAVE_EVERY          = 5
BATCH_SIZE_GPU_HIGH_VRAM = 8   # matches colab/voice_server.ipynb's BATCH_SIZE, tuned for a 16GB T4
BATCH_SIZE_GPU_LOW_VRAM  = 4   # cards under LOW_VRAM_THRESHOLD_GB (e.g. a 4GB laptop 3050) -- 8 risks CUDA OOM
LOW_VRAM_THRESHOLD_GB    = 10  # same cutoff colab/voice_server.ipynb's own GPU-check cell warns at
BATCH_SIZE_CPU      = 4
EPOCHS_GPU          = 200    # matches colab/voice_server.ipynb TOTAL_EPOCHS
EPOCHS_CPU          = 40     # CPU has no realistic path to 200 epochs -- bounded fallback instead

# Early stopping (CPU fallback only in practice -- GPU/Colab training is fast enough
# that EPOCHS_GPU rarely needs cutting short, but the logic isn't device-specific).
# RVC's generator loss (loss_gen + loss_fm + loss_mel + loss_kl, i.e. train.py's own
# "loss_gen_all" minus the discriminator's adversarial loss_disc, which reflects the
# discriminator's own state rather than output quality) is noisy epoch-to-epoch since
# this is adversarial (GAN) training, not a monotonically-decreasing supervised loss --
# so patience needs to be generous enough to ride out normal oscillation instead of
# bailing on a temporary bad epoch:
#   - EARLY_STOP_MIN_EPOCHS: no stopping before this many epochs -- the first several
#     epochs are the noisiest (generator and discriminator are still finding balance),
#     so convergence judgments there are unreliable.
#   - EARLY_STOP_PATIENCE: 5, not 10 (revised 2026-08-12) -- patience only counts
#     genuinely fresh loss readings (see the metric_is_fresh fix below), and in
#     practice a fresh reading only lands roughly every ~25 epochs, not every epoch.
#     At 10, patience could need up to ~250 epochs' worth of fresh-reading gaps to
#     exhaust -- past the 200-epoch cap entirely, i.e. early stopping effectively
#     never fired (confirmed for real: a run sat at loss 35.340 from epoch 51 through
#     at least epoch 72 with no sign of stopping). 5 was tried and rejected once before
#     under the *old* per-epoch counting (see git history), but that risk doesn't apply
#     here: MIN_EPOCHS=15 already skips the noisiest epochs, and each of the 5 patience
#     "strikes" is itself a real fresh comparison ~25 epochs apart, not 5 consecutive
#     noisy single epochs -- so 5 fresh non-improving readings is still riding out real
#     oscillation, just within a cap the run can actually reach.
#   - EARLY_STOP_MIN_DELTA: minimum loss decrease to count as "improvement" -- without
#     this, floating-point-noise-sized "improvements" would keep resetting the patience
#     counter forever.
EARLY_STOP_MIN_EPOCHS = 15
EARLY_STOP_PATIENCE    = 5
EARLY_STOP_MIN_DELTA   = 0.01

# If train.train produces *no output at all* (not even a single log line) for this
# long, treat it as hung rather than just slow, and kill it -- hit a real case of this:
# a run that stayed at 0% GPU utilization indefinitely, never getting far enough to
# print anything or touch the GPU (a Windows spawn/DataLoader-worker deadlock is the
# leading suspect, unconfirmed). Without this, a hang left the profile stuck at
# status='training' forever with no error, silently blocking every future retrain
# attempt (both the client-facing and manager-dashboard buttons refuse to fire while
# status=='training') -- this is exactly what happened before this constant existed.
# 10 minutes is generous enough to cover slow model loading / CUDA warmup on a weak
# GPU or first-ever CPU run without false-triggering on normal (if slow) progress.
STALL_TIMEOUT_SEC = 600

# Same stall concern as STALL_TIMEOUT_SEC above, applied to the venv dependency
# install (ensure_set_up()) -- a plain subprocess.run(..., check=True) with no
# timeout= blocks forever on a hung pip with no way to notice. Hit this for real:
# pip sat stuck for 37 minutes (one connection stuck in CLOSE_WAIT) with zero
# bytes written to site-packages the whole time, no error, just going nowhere.
# 15 minutes is generous for this file's larger wheels (onnxruntime, opencv,
# transformers, ...) even on a slow connection, while still failing well before
# a manager would give up waiting and assume something's broken.
PIP_INSTALL_TIMEOUT_SEC = 900

# Matches train/train.py's own per-step log line (see its logger.info(f"loss_disc=...
# loss_gen=... loss_fm=...loss_mel=... loss_kl=...")) -- \s* rather than a fixed space
# count since that f-string has inconsistent spacing around the commas.
_LOSS_LINE_RE = re.compile(
    r"loss_gen=([\d.]+),\s*loss_fm=([\d.]+),\s*loss_mel=([\d.]+),\s*loss_kl=([\d.]+)"
)
# train/train.py's per-epoch marker (i18n("====> ...").format(epoch, ...)) -- the
# "====> " prefix is part of the i18n *key* itself, kept as-is by every locale file
# checked (including en_US: "====> Epoch: {} {}"), so matching just the prefix reliably
# detects an epoch boundary regardless of which language train.py's logger is using.
_EPOCH_MARKER = "====> "

# convert_local() shells out to infer/cli.py per call (no cached in-process model --
# see its docstring for why), so this only needs to bound one subprocess's runtime.
CONVERT_TIMEOUT_SEC = 120


def device() -> str:
    """"cuda:0" if this machine has a usable GPU, else "cpu" -- the same
    detection Whisper already does internally in voice/stt.py, made explicit
    here since the RVC scripts (unlike whisper.load_model) need the device
    string passed in explicitly at several points."""
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
    assets are already present locally (no network access needed, no download
    triggered)."""
    return (os.path.isdir(RVC_REPO_DIR) and os.path.exists(_venv_python())
            and all(os.path.exists(p) for p in ASSETS))


def _venv_python() -> str:
    if os.name == "nt":
        return os.path.join(RVC_VENV_DIR, "Scripts", "python.exe")
    return os.path.join(RVC_VENV_DIR, "bin", "python")


# The cloned repo's own requirements files (requirments_{cpu,cu118,cu128}_py312.txt)
# are written for the full Gradio 3.14 webui + UVR5 vocal-separation tool -- ~40
# packages including gradio/fastapi/pydantic/onnxruntime/opencv/torch-directml/etc.
# Installing that whole file is what kept failing: a hash mismatch against its
# default Chinese mirror, then repeated network stalls on its largest wheels
# (onnxruntime, opencv, ...) even from official PyPI. None of that is actually
# needed -- an AST scan of every module the 5 training scripts we call actually
# import (train/preprocess.py, train/dataset/extract_f0.py,
# train/dataset/extract_hubert_feature.py, train/train.py, train/train_index.py,
# and their own local imports: train/data_utils.py, train/losses.py,
# train/mel_processing.py, train/process_ckpt.py, train/utils.py,
# train/dataset/slicer2.py, infer/hubert.py, infer/audio.py, infer/module/*,
# i18n/i18n.py, tools/*, configs/config.py) found only these third-party packages
# beyond what rag_env's own torch/numpy/scipy/librosa/soundfile/faiss already
# provide via --system-site-packages:
#   av, ffmpeg-python, matplotlib, praat-parselmouth, scikit-learn, transformers,
#   and tensorboard (train.py does `from torch.utils.tensorboard import
#   SummaryWriter` -- needs the standalone tensorboard package even though the
#   import statement starts with "torch.").
# Installing exactly these, from official PyPI, sidesteps every failure mode
# above at once rather than chasing them one at a time.
TRAINING_PACKAGES = [
    "av", "ffmpeg-python", "matplotlib", "praat-parselmouth",
    "scikit-learn", "tensorboard", "transformers",
]


SETUP_LOCK_TIMEOUT_SEC = 1800  # generous -- a legitimate first-run setup can itself take a while


def _acquire_setup_lock(report) -> int:
    """Cross-process mutex around ensure_set_up()'s repo-clone/venv-build/asset-
    download sequence. Needed because two run_training() calls for the same (or
    even different) speaker can genuinely happen concurrently -- e.g. a manager
    clicking retrain while an earlier attempt is still running -- and without
    this, both processes share the same on-disk local_rvc/ directory tree with
    no coordination. Hit this for real: one process's rebuild (rmtree, since its
    deps marker didn't match) deleted the venv while another process's pip
    install was actively mid-download inside it, corrupting both runs at once.

    Returns an open file descriptor to release via _release_setup_lock() when
    done. A stale lock (owner crashed/was killed without cleaning up) is
    reclaimed after SETUP_LOCK_TIMEOUT_SEC rather than blocking forever.
    """
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
                age = SETUP_LOCK_TIMEOUT_SEC + 1  # lock vanished mid-check -- treat as stale, retry the open
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
    """Clones RVC-Project's WebUI repo, builds an isolated venv for it, and
    downloads the pretrained assets training needs (same source as
    colab/voice_server.ipynb cells C/E). Only does what's missing, so it's
    cheap to call before every local train/convert attempt once everything is
    already in place. Raises on failure (network, git, disk space, or a pip
    install failure) -- callers should treat that as "local fallback
    unavailable right now".

    The venv is required because this repo's own pinned dependencies
    (fastapi<0.100, pydantic<2, starlette<0.28 -- it still ships a Gradio 3.14
    webui we never use) directly conflict with clone-voice-station's own
    FastAPI stack. Installing them into this process's environment would
    downgrade/break the running app, so training subprocesses always use
    _venv_python() instead of sys.executable. Local CONVERSION (rvc-python, in
    convert_local() above) is unaffected -- it's a separate lightweight
    package with no such conflict, so it keeps running in-process.

    Serialized across processes via _acquire_setup_lock() -- see its docstring
    for why (a real corruption from two concurrent callers racing on the same
    on-disk venv).
    """
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

    # Tracked inside the venv dir itself (not the repo dir) so a previous attempt
    # that cloned the repo but died mid pip-install doesn't get permanently skipped
    # on retry just because the repo folder already exists. No longer needs a
    # variant suffix: with --system-site-packages (below), the venv always sees
    # whatever torch build rag_env currently has live, automatically -- there's no
    # separate torch install to fall out of sync with GPU/CPU status anymore, so a
    # rebuild is never needed just because this process gained/lost CUDA.
    deps_marker = os.path.join(RVC_VENV_DIR, ".deps_installed")
    if not os.path.exists(deps_marker):
        if os.path.isdir(RVC_VENV_DIR):
            shutil.rmtree(RVC_VENV_DIR, ignore_errors=True)
        report("Creating isolated venv for RVC training (first run only)…")
        # --system-site-packages: lets the venv see this process's own already-installed
        # packages (torch above all) instead of needing its own separate copy -- avoids
        # downloading a second, redundant torch+torchaudio pair (~2-3GB) when this
        # process's own torch (rag_env's) is already confirmed working. Packages the
        # venv installs itself (below) still shadow/override the system ones for
        # anything run inside the venv, so the conflicting pins this venv exists to
        # isolate (fastapi<0.100, pydantic<2, gradio, etc. -- see ensure_set_up's
        # docstring) still can't leak into or downgrade rag_env's own FastAPI stack.
        subprocess.run([sys.executable, "-m", "venv", "--system-site-packages", RVC_VENV_DIR],
                       check=True, timeout=120)

        report(f"Installing training dependencies into isolated venv "
               f"({', '.join(TRAINING_PACKAGES)})…")
        try:
            subprocess.run(
                [_venv_python(), "-m", "pip", "install", "-q",
                 "--index-url", "https://pypi.org/simple",
                 # pip's default socket timeout (15s) is too short for these wheels over
                 # a plain PyPI download -- hit a real ReadTimeoutError on
                 # files.pythonhosted.org mid-download. --retries adds resilience
                 # against otherwise-transient drops on top of the longer timeout.
                 "--timeout", "120", "--retries", "5",
                 # A stalled install here has needed a hard kill more than once (see
                 # PIP_INSTALL_TIMEOUT_SEC/STALL_TIMEOUT_SEC) -- killing pip mid-write
                 # can leave a truncated file in its local cache, which a *later*
                 # attempt then reuses and flags as a hash mismatch ("may have been
                 # tampered with") even though nothing external is actually wrong.
                 # Hit exactly this. --no-cache-dir avoids reusing anything from a
                 # previous, possibly-interrupted run.
                 "--no-cache-dir",
                 *TRAINING_PACKAGES],
                check=True, timeout=PIP_INSTALL_TIMEOUT_SEC,
            )
        except subprocess.TimeoutExpired:
            # --timeout/--retries above only bound individual socket reads, not the
            # whole command -- hit a real case of this too: pip sat for 37 minutes
            # with a connection stuck in CLOSE_WAIT and zero bytes written to
            # site-packages the entire time, no error, just silently going nowhere.
            # subprocess.run's own timeout= is the only thing that bounds the full
            # invocation.
            raise RuntimeError(
                f"pip install stalled for over {PIP_INSTALL_TIMEOUT_SEC // 60} minutes with no "
                f"progress (not a normal slow-but-working download) and was killed."
            )
        open(deps_marker, "w").close()
        report("Training dependencies installed.")

    # faiss-cpu needs to be in *this* process's site-packages, not just installed
    # somewhere -- the venv is --system-site-packages (inherits from here), and
    # infer/cli.py (convert_local() above) needs faiss itself for the
    # retrieval-index blend at inference time, same as train/train_index.py does
    # for training.
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
    # Same voice_storage/<speaker_id>/<speaker_id>.{pth,index} layout
    # engine/voice_engine.py's _download_and_store_model() already uses for the
    # Colab-trained backup copy, so a locally-trained model is indistinguishable
    # from a downloaded one to every other caller.
    model_dir = os.path.join(VOICE_MODELS_DIR, speaker_id)
    return os.path.join(model_dir, f"{speaker_id}.pth"), os.path.join(model_dir, f"{speaker_id}.index")


def has_local_model(speaker_id: str) -> bool:
    pth_path, index_path = _model_paths(speaker_id)
    return os.path.exists(pth_path) and os.path.exists(index_path)


def convert_local(audio_bytes: bytes, speaker_id: str, pitch: int = None,
                   index_rate: float = None, mime: str = "audio/mp3") -> bytes | None:
    """
    Runs voice conversion using the speaker's locally-available model
    (voice_storage/<speaker_id>/ -- either downloaded from Colab after a prior
    remote train, or produced by train_speaker_local() below).

    Originally used the rvc-python PyPI package's in-process RVCInference, but
    that requires fairseq==0.12.2, which cannot even be imported on Python 3.12
    -- a hard language-level break, not a missing-package problem: fairseq's
    dataclass definitions use a mutable-default pattern
    ("field: SomeDataclass = SomeDataclass()") that Python 3.11+'s dataclasses
    module now correctly rejects at class-definition time. No amount of pip
    installing works around that.

    Uses infer/cli.py (in the cloned repo, run inside the training venv)
    instead -- the same modern, fairseq-free pipeline (Transformers-based
    HuBERT, see infer/hubert.py) already proven working for training, so no
    separate setup or dependency story from that. Slower per-call than a
    cached in-process model would be (reloads the model fresh every time,
    since each call is its own subprocess) but this is already a fallback
    path only used when Colab is unavailable, so simplicity and not
    depending on fairseq wins over shaving off a load time.

    Returns None if no local model is available, or on any failure -- callers
    (voice/rvc_client.py) should treat that the same as "Colab unreachable and
    no local model either": fall back to unconverted TTS audio.
    """
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
    out_path = in_path + "_rvc.wav"

    try:
        # PYTHONUTF8=1: infer/cli.py prints some i18n status text containing non-Latin
        # characters (e.g. "【...】"-style brackets); without this, the child's stdout
        # defaults to the Windows console's codepage (cp1252 here), and printing one of
        # those characters crashes it with UnicodeEncodeError mid-run -- hit this for
        # real, right after the model had already loaded and inference had started.
        child_env = os.environ.copy()
        child_env["PYTHONUTF8"] = "1"
        result = subprocess.run([
            _venv_python(), "-m", "infer.cli",
            "--model", pth_path,
            "--index", index_path,
            "--input", in_path,
            "--output", out_path,
            "--pitch", str(pitch),
            "--f0-method", F0_METHOD,
            "--index-rate", str(index_rate),
            "--protect", str(PROTECT),
            "--overwrite",
        ], cwd=RVC_REPO_DIR, capture_output=True, timeout=CONVERT_TIMEOUT_SEC, env=child_env,
           # encoding=utf-8 (not text=True, which decodes with the locale default --
           # cp1252 here) so *this* side decodes the child's output consistently with
           # PYTHONUTF8=1 above. Without this, a background reader thread inside
           # subprocess.run() hit a UnicodeDecodeError of its own trying to decode the
           # same non-Latin bytes as cp1252 -- didn't break this particular run (the
           # audio file was already fully written by the time it happened) but printed
           # a scary, misleading traceback and could lose the tail of stderr on a run
           # where the timing worked out worse.
           encoding="utf-8", errors="replace")
        if result.returncode != 0:
            detail = (result.stdout[-800:] + "\n" + result.stderr[-800:]).strip()
            logger.error(f"[RVC-local] convert failed for {speaker_id} on {device()}:\n{detail}")
            return None
        with open(out_path, "rb") as f:
            return f.read()
    except subprocess.TimeoutExpired:
        logger.error(f"[RVC-local] convert timed out for {speaker_id} after {CONVERT_TIMEOUT_SEC}s")
        return None
    except Exception as e:
        logger.error(f"[RVC-local] convert failed for {speaker_id} on {device()}: {e}")
        return None
    finally:
        for p in (in_path, out_path):
            if os.path.exists(p):
                os.unlink(p)


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
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if result.returncode != 0:
        # These scripts often catch their own exceptions and log a "[X][Failed] ..."
        # line to stdout (or a log file) instead of letting a traceback hit stderr --
        # hit this for real (extract_hubert_feature.py's "Model not found" case
        # printed nothing to stderr at all) -- so surface both, not just stderr.
        detail = (result.stdout[-1000:] + "\n" + result.stderr[-1000:]).strip()
        raise RuntimeError(f"{label} failed (exit {result.returncode}):\n{detail}")
    return result.stdout


def _write_train_config(exp_dir: str):
    """train/utils.py's get_hparams() does
    `json.loads(read_text(os.path.join(experiment_dir, "config.json")))` with no
    fallback -- FileNotFoundError if it's missing. webui.py's click_train() writes
    this itself (copying a template from configs/) before ever invoking train.py;
    since we call train.train directly, we have to do the same. v2/40k.json doesn't
    exist in this repo (only v1/{32k,40k,48k} and v2/{32k,48k}) -- v1/40k.json is
    the correct template regardless of RVC_VERSION, matching webui.py's own
    "if version == 'v1' or sr == '40k': use v1" selection (SAMPLE_RATE is fixed at
    40k here, so this is always that case)."""
    src = os.path.join(RVC_REPO_DIR, "configs", "v1", "40k.json")
    shutil.copy2(src, os.path.join(exp_dir, "config.json"))


def _run_training_with_early_stop(venv_python: str, exp_dir: str, speaker_id: str,
                                    batch_size: int, epochs: int, report) -> int:
    """
    Runs train/train.py, watching its live output so training can stop early once
    the generator loss (loss_gen + loss_fm + loss_mel + loss_kl -- train.py's own
    "loss_gen_all" minus the discriminator's own adversarial loss_disc, which
    reflects the discriminator's state rather than output quality) hasn't improved
    for EARLY_STOP_PATIENCE epochs, instead of always running the full epoch count.
    See the EARLY_STOP_* constants above for why the numbers are what they are.

    Returns the last epoch number actually reached (< epochs if stopped early).
    Raises RuntimeError if the subprocess fails for a reason other than our own
    early-stop termination.
    """
    # -c 1 (if_cache_data_in_gpu) loads the whole dataset into VRAM once instead of
    # transferring a fresh batch from CPU every step -- on a small personal-voice
    # dataset (a few minutes of audio, a few dozen segments) this comfortably fits
    # even on a 4GB card, and removes the main reason GPU utilization stays low on
    # a workload this size (CPU-side data loading becoming the bottleneck between
    # kernel launches). Left at 0 on CPU, where there's no CPU<->GPU transfer to cache.
    cache_in_gpu = "1" if is_gpu() else "0"
    proc = subprocess.Popen([
        venv_python, "-m", "train.train",
        "-e", speaker_id, "-sr", "40k", "-f0", "1", "-bs", str(batch_size),
        "-g", "0", "-te", str(epochs), "-se", str(SAVE_EVERY),
        "-pg", os.path.join("assets", "pretrained_v2", "f0G40k.pth"),
        "-pd", os.path.join("assets", "pretrained_v2", "f0D40k.pth"),
        "-l", "1", "-c", cache_in_gpu, "-sw", "0", "-v", RVC_VERSION,
    ], cwd=RVC_REPO_DIR, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)

    # `for line in proc.stdout` blocks with no timeout, so a hung child (rather than a
    # crashed one) would block this forever with no way to notice -- hit this for real:
    # a run that sat at 0% GPU utilization indefinitely, never printing a single line.
    # Reading on a background thread and pulling from a queue with a timeout is the
    # standard way to get a timeout on a blocking pipe read in Python (no cross-platform
    # way to put a timeout directly on the read itself).
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
            # train.py logs a loss line every train.log_interval *steps* (200 by
            # default), not every epoch -- on a small personal-voice dataset (a few
            # dozen segments, ~14 steps/epoch here) a fresh reading can be 10+ epochs
            # apart. Evaluating improve/no-improve on every epoch regardless (the
            # earlier version of this code) meant no_improve was really counting
            # "epochs since the last log line", not "epochs since the last real
            # non-improvement" -- hit this for real: patience=10 combined with a
            # ~14-epoch gap between readings triggered an early stop at epoch 17
            # after the *same* stale loss value got re-evaluated 11 times in a row.
            # Only counting epochs with a genuinely fresh reading fixes that.
            if metric_is_fresh:
                last_epoch = epoch
                if epoch < EARLY_STOP_MIN_EPOCHS:
                    # Readings before MIN_EPOCHS don't touch best_metric/no_improve at all --
                    # not just "can't trigger a stop yet" but "never counted as a strike" --
                    # so a few noisy non-improving early readings can't pre-load patience and
                    # cause an immediate stop the instant epoch crosses MIN_EPOCHS with no real
                    # post-MIN_EPOCHS evaluation. The first eligible reading (epoch >=
                    # MIN_EPOCHS) becomes the baseline "best" fresh, same as if training started
                    # there.
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
                # last_epoch (not best_epoch) here -- last_metric is the most recent fresh
                # reading, which isn't the best one once a non-improving reading comes in
                # (e.g. a 39.174 reading at epoch 76 after a 35.340 best @ epoch 51 previously
                # showed the misleading "last: 39.174 @ epoch 51", pairing a later value with
                # an earlier, unrelated epoch number).
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
    """
    Builds assets/weights/<speaker_id>.pth from the latest raw training checkpoint
    when early stopping cut the run short. train/train.py's own savee() call (see
    train/process_ckpt.py) only fires once epoch >= the full -te target, which an
    early-stopped run never reaches, so we have to reproduce that step ourselves.
    Runs inside the training venv (not this process) since it needs the same
    torch/train.* imports training itself used.
    """
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
        # savee() catches its own exceptions and returns the traceback as a string
        # rather than raising -- see train/process_ckpt.py.
        raise RuntimeError(f"savee() failed while finalizing early-stopped model:\n{result.stdout[-1500:]}")


def train_speaker_local(speaker_id: str, sample_files: list, progress_cb=None) -> tuple[str, str]:
    """
    Full local training pipeline for one speaker: slice/normalize -> preprocess
    -> F0 -> HuBERT features -> train RVC v2 -> FAISS index -> save into
    voice_storage/<speaker_id>/.

    Calls the same underlying scripts webui.py itself shells out to (confirmed
    by reading its run_preprocess_dataset/run_extract_f0_feature/click_train/
    run_train_index functions directly, since the upstream repo was restructured
    at some point after colab/voice_server.ipynb was written -- the old flat
    trainset_preprocess_pipeline_print.py/extract_f0_print.py/
    extract_feature_print.py/train.py no longer exist; they moved to
    train/preprocess.py, train/dataset/extract_f0.py,
    train/dataset/extract_hubert_feature.py, train/train.py, with a new
    train/train_index.py replacing the hand-rolled FAISS step this function
    used to do itself). All run inside the isolated venv from ensure_set_up(),
    not this process's own interpreter.

    sample_files: list of (filename, bytes) tuples -- same shape
    engine/voice_engine.py already builds from list_voice_samples() for the
    Colab path, so callers don't need a separate local variant.

    Raises RuntimeError (or lets a subprocess CalledProcessError propagate) on
    failure -- the caller (engine/voice_engine.py's run_training) is
    responsible for turning that into a "failed" profile status, same as it
    already does for a failed Colab job.
    """
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
        # savee() (train/process_ckpt.py) writes to the hardcoded relative path
        # "assets/weights/<name>.pth" and doesn't create its parent dir -- this repo
        # doesn't ship that folder, so without this a fresh clone fails here on *any*
        # completed run, not just an early-stopped one (hit this for real: it broke
        # the early-stop finalize path first, but train.train's own internal savee()
        # call on natural completion would hit the exact same error).
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
        # Invoked as "-m train.preprocess", NOT "train/preprocess.py" -- running it as
        # a bare file path puts the script's own directory (.../RVC/train) at the front
        # of sys.path, which collides with the sibling file train/train.py (also named
        # "train"): "from train.dataset.slicer2 import Slicer" then resolves "train" to
        # that *file* instead of the package, and train.py's own "from train import
        # utils" recurses into itself -> circular-import crash (hit this for real on a
        # live run). "-m" puts cwd (=RVC_REPO_DIR) on sys.path instead, matching how
        # webui.py itself resolves these same imports when it's the process entry point.
        _run_step([venv_python, "-m", "train.preprocess",
                   sliced_dir, str(SAMPLE_RATE), str(n_p), exp_dir, "False", "3.7"],
                  "Preprocessing", cwd=RVC_REPO_DIR)

        report("Extracting F0 (RMVPE)…")
        # "cpu" mode regardless of dev: webui.py itself only uses the cuda-parallel
        # form when a specific gpu list is configured for rmvpe; its own default
        # (no such config) is this same single-process cpu path.
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

        # train/utils.py's get_hparams() unconditionally reads {exp_dir}/config.json
        # with no fallback -- normally written by webui.py's own click_train() before
        # it invokes train.py, a step this port has to replicate since train.train is
        # called directly here.
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

        # train/train.py only saves the finished model to assets/weights/<name>.pth
        # (see train/process_ckpt.py's savee()) when it reaches the full -te epoch
        # target -- an early-stopped run never gets there, so build it ourselves from
        # the last raw checkpoint in that case.
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
