"""
voice/rvc_client.py
Client for the RVC voice-conversion + training service. Primary backend is
the server that runs on Google Colab (see colab/voice_server.ipynb), reached
over HTTP through a cloudflared tunnel. When that endpoint is unset or
unreachable, convert() and train_local() (used by engine/voice_engine.py) fall
back to voice/rvc_local.py, which runs the same pipeline on this machine's own
GPU (if present) or CPU — so voice cloning keeps working without a Colab
session running, just slower and (for training) at lower quality on CPU.

The tunnel URL and every tuning knob below (pitch, index_rate, timeouts) are
read from the settings DB table — editable live from the manager dashboard
(GET/POST /manager/rvc_config, see app.py) — so a manager can retune the
connection without restarting the service. The RVC_* env vars are only used
as the seed default the first time each setting is read (see database.database
.get_setting's `default` param below) and as a fallback if the DB is somehow
unreachable.

All functions degrade gracefully (return "unavailable"/original audio) when
the endpoint is unset or unreachable *and* the local fallback also has no
usable model — voice playback must keep working with plain TTS even then.
"""

import os
import requests

from engine.server_log import get_logger

logger = get_logger()

from database.database import get_setting

# Defaults used only the first time a setting is read (no row in the DB yet) —
# after that, whatever's stored in `settings` (editable via the manager
# dashboard) always wins. See get_pitch()/get_index_rate()/get_timeout().
_DEFAULTS = {
    # 20s was tuned for a fast Colab T4 response and is far too short once
    # `rvc_endpoint` points at local_rvc_server.py instead (see its own docstring
    # and voice/rvc_local.py's CONVERT_TIMEOUT_SEC comment): its first call per
    # speaker legitimately reloads the model from disk (measured 101s cold, 0.8s
    # warm after). At 20s, that cold call always looked like "endpoint
    # unreachable" and fell through to the even-slower raw subprocess fallback,
    # which then *also* timed out -- confirmed for real via repeated "[RVC]
    # Endpoint unreachable ... Read timed out (read timeout=20)" log lines.
    # 150s matches CONVERT_TIMEOUT_SEC's own margin and is harmless for genuine
    # Colab use too (only makes a slow-but-working Colab response less likely to
    # be abandoned early, never a regression).
    "rvc_timeout_convert":    os.getenv("RVC_TIMEOUT_CONVERT", "150"),    # seconds
    "rvc_timeout_short":      os.getenv("RVC_TIMEOUT_SHORT", "5"),        # health/status checks
    "rvc_timeout_download":   os.getenv("RVC_TIMEOUT_DOWNLOAD", "180"),   # trained model can be 50-150MB
    "rvc_timeout_f5tts":      os.getenv("RVC_TIMEOUT_F5TTS", "30"),       # F5-TTS is slower than RVC convert
    "rvc_timeout_transcribe": os.getenv("RVC_TIMEOUT_TRANSCRIBE", "20"),  # PhoWhisper-large on a T4, ~real-time
    "rvc_pitch":              os.getenv("RVC_PITCH", "0"),
    "rvc_index_rate":         os.getenv("RVC_INDEX_RATE", "0.75"),
}


def get_endpoint() -> str:
    return (get_setting("rvc_endpoint") or os.getenv("RVC_ENDPOINT", "")).rstrip("/")


def get_pitch() -> int:
    return int(get_setting("rvc_pitch", _DEFAULTS["rvc_pitch"]))


def get_index_rate() -> float:
    return float(get_setting("rvc_index_rate", _DEFAULTS["rvc_index_rate"]))


def get_timeout(name: str) -> int:
    """name: one of 'convert' | 'short' | 'download' | 'f5tts' | 'transcribe'."""
    key = f"rvc_timeout_{name}"
    return int(get_setting(key, _DEFAULTS[key]))


def is_available() -> bool:
    endpoint = get_endpoint()
    if not endpoint:
        return False
    try:
        r = requests.get(f"{endpoint}/health", timeout=get_timeout("short"))
        return r.status_code == 200
    except requests.exceptions.RequestException:
        return False


def convert(audio_bytes: bytes, speaker_id: str, pitch: int = None,
            index_rate: float = None, mime: str = "audio/mp3") -> bytes:
    """
    Sends TTS audio for conversion into the given speaker's cloned voice —
    tries the Colab RVC server first, then this machine's own GPU/CPU
    (voice/rvc_local.py) if Colab is unset/unreachable/errors. Falls back to
    the original (unconverted) audio if neither backend has a usable model
    for this speaker yet.
    """
    endpoint = get_endpoint()
    pitch      = get_pitch() if pitch is None else pitch
    index_rate = get_index_rate() if index_rate is None else index_rate

    if endpoint:
        try:
            resp = requests.post(
                f"{endpoint}/convert",
                files={"audio": ("input.mp3", audio_bytes, mime)},
                data={"speaker_id": speaker_id, "pitch": pitch, "index_rate": index_rate},
                timeout=get_timeout("convert"),
            )
            if resp.status_code == 200:
                return resp.content
            logger.warning(f"[RVC] /convert returned {resp.status_code} — trying local fallback.")
        except requests.exceptions.RequestException as e:
            logger.warning(f"[RVC] Endpoint unreachable ({e}) — trying local fallback.")

    local_result = _convert_local(audio_bytes, speaker_id, pitch, index_rate, mime)
    if local_result is not None:
        return local_result

    logger.warning(f"[RVC] No Colab and no local model for {speaker_id} — falling back to TTS audio.")
    return audio_bytes


def _convert_local(audio_bytes: bytes, speaker_id: str, pitch: int,
                    index_rate: float, mime: str) -> bytes | None:
    """Local import: keeps voice/rvc_local.py (and its torch/rvc-python
    dependency) optional for a deployment that always has Colab available."""
    try:
        from voice import rvc_local
        return rvc_local.convert_local(audio_bytes, speaker_id, pitch=pitch,
                                        index_rate=index_rate, mime=mime)
    except Exception as e:
        logger.warning(f"[RVC] Local fallback unavailable for {speaker_id}: {e}")
        return None


def train_local(speaker_id: str, samples: list, progress_cb=None) -> tuple[str, str]:
    """
    Runs RVC training on this machine's own GPU (if present) or CPU instead of
    Colab — see voice/rvc_local.py for the actual pipeline. Used by
    engine/voice_engine.py's run_training() when start_train() reports Colab
    is unset/unreachable. Raises on failure; caller is responsible for turning
    that into a "failed" profile status, same as a failed Colab job.
    """
    from voice import rvc_local
    return rvc_local.train_speaker_local(speaker_id, samples, progress_cb=progress_cb)


def start_train(speaker_id: str, samples: list) -> dict:
    """
    Uploads training samples and asks the Colab server to enqueue a training
    job. Returns immediately (does not wait for training to finish) — poll
    train_status() to track progress.

    samples: list of (filename, bytes) tuples.
    """
    endpoint = get_endpoint()
    if not endpoint:
        return {"status": "unavailable", "message": "Chưa cấu hình RVC endpoint (Colab)."}

    try:
        files = [("files", (fname, data, "audio/wav")) for fname, data in samples]
        resp = requests.post(
            f"{endpoint}/train",
            data={"speaker_id": speaker_id},
            files=files,
            timeout=60,
        )
        if resp.status_code == 200:
            return resp.json()
        return {"status": "error", "message": f"Colab trả về lỗi HTTP {resp.status_code}"}
    except requests.exceptions.RequestException as e:
        return {"status": "error", "message": f"Không kết nối được tới Colab: {e}"}


# Cloudflare's own edge error codes for a tunnel that's up but not currently answering
# (origin/cloudflared unresponsive) -- confirmed for real: a live training run got HTTP 530
# ("Cloudflare Tunnel error 1033") back from a poll a few minutes after entering the heavy
# GPU training phase, while training kept running on Colab regardless. Unlike a genuine
# app-level error (Flask itself returning 400/404/500), these come from Cloudflare's edge
# *instead of* reaching our server at all, so they get the same "still might be running,
# don't give up immediately" leniency as a network-level RequestException below, not the
# immediate-failure treatment a real HTTP error from our own app gets.
_CLOUDFLARE_TUNNEL_ERROR_CODES = {502, 503, 504, 520, 521, 522, 523, 524, 525, 526, 527, 530}


def train_status(speaker_id: str) -> dict:
    endpoint = get_endpoint()
    if not endpoint:
        return {"status": "unavailable"}

    try:
        resp = requests.get(f"{endpoint}/train_status/{speaker_id}", timeout=get_timeout("short"))
        if resp.status_code == 200:
            return resp.json()
        if resp.status_code in _CLOUDFLARE_TUNNEL_ERROR_CODES:
            return {"status": "network_error", "message": f"Cloudflare Tunnel error (HTTP {resp.status_code})"}
        return {"status": "error", "message": f"HTTP {resp.status_code}"}
    except requests.exceptions.RequestException as e:
        # Distinct from "error" above -- confirmed for real that a real training run (heavy
        # GPU work occupying the Colab VM) can make cloudflared itself briefly unresponsive
        # (a "Cloudflare Tunnel error 1033" page instead of a response) well within this
        # call's own 5s timeout, even though training keeps running on Colab regardless of
        # whether this one poll got through. _poll_until_done() tolerates a run of these
        # before giving up; "error" (a real HTTP response, just not 200) and "unavailable"
        # (no endpoint configured at all) still fail immediately, same as before -- only a
        # network-level failure to even reach the endpoint gets this leniency.
        return {"status": "network_error", "message": str(e)}


def download_model(speaker_id: str) -> bytes:
    """
    Downloads the trained model (.pth + .index, zipped) for this speaker from
    Colab so the app can keep its own local backup copy. Returns None if the
    endpoint is unset/unreachable or no trained model exists yet — callers
    should treat that as "no local backup available", not a hard failure,
    since the voice still works via Colab's own Drive-stored copy either way.
    """
    endpoint = get_endpoint()
    if not endpoint:
        return None

    try:
        resp = requests.get(f"{endpoint}/models/{speaker_id}/download", timeout=get_timeout("download"))
        if resp.status_code == 200:
            return resp.content
        logger.warning(f"[RVC] /models/{speaker_id}/download returned {resp.status_code}")
        return None
    except requests.exceptions.RequestException as e:
        logger.warning(f"[RVC] Model download failed for {speaker_id}: {e}")
        return None


def synthesize_f5tts(gen_text: str, ref_audio_bytes: bytes = None, ref_text: str = None,
                      speed: float = 1.0) -> bytes:
    """
    Calls the F5-TTS-Vietnamese-ViVoice baseline endpoint on Colab (see
    colab/voice_server.ipynb, "Baseline: F5-TTS-Vietnamese-ViVoice" section). Used both
    as the RQ2 zero-shot evaluation baseline and as an optional base-voice engine (see
    voice/tts.py) selected via a profile's base_tts_voice = "f5tts:default".

    ref_audio_bytes/ref_text are optional — omit them to use the bundled reference clip
    configured on the Colab side (REF_AUDIO_F5TTS / REF_TEXT_F5TTS).

    Returns None if the endpoint is unset/unreachable/errors — callers must fall back to
    edge-TTS in that case, same as convert() falls back to unconverted audio.
    """
    endpoint = get_endpoint()
    if not endpoint:
        return None

    files = {}
    if ref_audio_bytes is not None:
        files["ref_audio"] = ("ref.wav", ref_audio_bytes, "audio/wav")
    data = {"gen_text": gen_text, "speed": speed}
    if ref_text is not None:
        data["ref_text"] = ref_text

    try:
        resp = requests.post(
            f"{endpoint}/baseline/f5tts", files=files, data=data, timeout=get_timeout("f5tts"),
        )
        if resp.status_code == 200:
            return resp.content
        logger.warning(f"[F5-TTS] /baseline/f5tts returned {resp.status_code} — falling back to edge-TTS.")
        return None
    except requests.exceptions.RequestException as e:
        logger.warning(f"[F5-TTS] Endpoint unreachable ({e}) — falling back to edge-TTS.")
        return None


def transcribe_remote(audio_bytes: bytes, mime: str = "audio/webm", language: str = None) -> dict:
    """
    Calls the PhoWhisper (Vietnamese-tuned) STT endpoint on Colab (see
    colab/voice_server.ipynb, "Speech-to-Text: PhoWhisper" section) — the primary ASR
    path. Returns None if the endpoint is unset/unreachable/errors, same fallback
    contract as synthesize_f5tts(): callers must fall back to the local, CPU-only
    openai-whisper model in voice/stt.py in that case.
    """
    endpoint = get_endpoint()
    if not endpoint:
        return None

    data = {"language": language} if language else {}
    try:
        resp = requests.post(
            f"{endpoint}/transcribe",
            files={"audio": ("input.webm", audio_bytes, mime)},
            data=data,
            timeout=get_timeout("transcribe"),
        )
        if resp.status_code == 200:
            return resp.json()
        logger.warning(f"[STT] /transcribe returned {resp.status_code} — falling back to local Whisper.")
        return None
    except requests.exceptions.RequestException as e:
        logger.warning(f"[STT] Endpoint unreachable ({e}) — falling back to local Whisper.")
        return None


def delete_model(speaker_id: str) -> dict:
    """
    Deletes the trained .pth/.index files for this speaker from the Colab
    server's Drive storage (and evicts it from the in-memory inference cache).
    Best-effort: callers should still remove the local DB/profile row even if
    this reports "unavailable" — otherwise a deleted user profile could never
    be cleaned up while Colab happens to be offline.
    """
    endpoint = get_endpoint()
    if not endpoint:
        return {"status": "unavailable", "message": "Chưa cấu hình RVC endpoint (Colab)."}

    try:
        resp = requests.delete(f"{endpoint}/models/{speaker_id}", timeout=get_timeout("short"))
        if resp.status_code in (200, 404):
            return resp.json() if resp.content else {"status": "ok"}
        return {"status": "error", "message": f"HTTP {resp.status_code}"}
    except requests.exceptions.RequestException as e:
        return {"status": "error", "message": f"Không kết nối được tới Colab: {e}"}
