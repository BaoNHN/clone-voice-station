"""
engine/voice_engine.py
Orchestrates the two voice features this service exposes to its clients:

  1. speak_text()   — TTS (+ RVC conversion, if the profile has a ready
                       cloned voice) for a client app's "read aloud" feature.
                       When RVC conversion is actually applied, the output is
                       prefixed with a spoken AI-disclosure notice and tagged
                       with a provenance watermark — required for audio that
                       simulates a real person's voice under Vietnam's Law on
                       Artificial Intelligence (134/2025/QH15, effective 1 Mar
                       2026), see _add_ai_disclosure() below.
  2. run_training()  — background job (called via FastAPI BackgroundTasks)
                       that uploads a profile's recorded samples to the Colab
                       RVC server and polls until the trained model is ready.
                       Falls back to training on this machine's own GPU/CPU
                       (voice/rvc_local.py, via rvc_client.train_local()) when
                       Colab is unset/unreachable.

Profile status lives in the voice_profiles DB row (not a separate in-memory
job store) so a client app can just poll GET /api/profiles/{id}/status.
"""

import asyncio
import io
import os
import time
import zipfile

from pydub import AudioSegment

from voice import tts, rvc_client
from database.database import (
    get_voice_profile, list_voice_samples, update_voice_profile_status, VOICE_MODELS_DIR,
)
from engine.server_log import get_logger

logger = get_logger()

# Bundled static ffmpeg (see bin/, gitignored) — same requirement/workaround as
# voice/stt.py: the conda-forge ffmpeg in rag_env fails to launch on this machine
# (STATUS_ENTRYPOINT_NOT_FOUND), which made pydub silently skip writing the
# AI-disclosure watermark tag in _add_ai_disclosure() below (pydub logs "Couldn't
# find ffmpeg or avconv" and falls back to a no-op exporter that drops tags).
# Falls back to PATH's ffmpeg if the bundled binary isn't present.
_BUNDLED_FFMPEG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bin")
if os.path.isfile(os.path.join(_BUNDLED_FFMPEG_DIR, "ffmpeg.exe")):
    os.environ["PATH"] = _BUNDLED_FFMPEG_DIR + os.pathsep + os.environ.get("PATH", "")

POLL_INTERVAL_SEC = 15
# Tolerance for transient network failures while polling (see rvc_client.train_status()'s
# "network_error" status) before giving up on an otherwise-still-running job -- confirmed
# for real that heavy GPU training work on the Colab VM can make cloudflared itself briefly
# unresponsive well within a single poll's own timeout, unrelated to whether training is
# actually still progressing. 20 * POLL_INTERVAL_SEC = 5 minutes of sustained
# unreachability, comfortably past what a compute-load blip has been observed to cause,
# while still giving up in finite time if the Colab session is genuinely gone for good.
MAX_CONSECUTIVE_NETWORK_ERRORS = 20
MAX_TRAIN_WAIT_SEC = 2 * 60 * 60  # 2 hours

# Vietnam's AI Law (134/2025/QH15) requires that audio simulating/impersonating a real
# person's voice carry a voice notice at the beginning identifying it as AI-generated.
# Only applies to actual RVC-cloned output — plain builtin TTS voices (HoaiMy, NamMinh,
# Jenny) aren't modeled on any specific real identifiable person, so they're out of scope.
# The clip is cached per base voice (module-level, process lifetime): fixed text, same
# TTS engine already used for the real answer, so there's nothing to regenerate per call.
AI_DISCLOSURE_TEXT_VI = (
    "Nội dung sau đây là giọng nói được tạo ra bởi trí tuệ nhân tạo, không phải giọng nói thật."
)
# Metadata-level provenance tag embedded in the output WAV, same class of approach as
# C2PA content credentials for images — not a robust/steganographic audio watermark,
# but a good-faith implementation of the law's watermark requirement for a prototype.
WATERMARK_TAG = "AI-generated / voice-converted audio - clone-voice-station RVC pipeline"

_disclosure_cache = {}


async def _get_disclosure_clip(base_voice: str) -> AudioSegment:
    if base_voice not in _disclosure_cache:
        audio_bytes, mime = await tts.synthesize(AI_DISCLOSURE_TEXT_VI, voice=base_voice)
        fmt = "mp3" if "mpeg" in mime else "wav"
        _disclosure_cache[base_voice] = AudioSegment.from_file(io.BytesIO(audio_bytes), format=fmt)
    return _disclosure_cache[base_voice]


async def _add_ai_disclosure(rvc_wav: bytes, base_voice: str) -> bytes:
    """Prepends the spoken AI-disclosure notice to RVC-converted audio and embeds the
    provenance tag above into the output WAV's metadata."""
    disclosure = await _get_disclosure_clip(base_voice)
    content    = AudioSegment.from_file(io.BytesIO(rvc_wav), format="wav")
    combined   = disclosure + AudioSegment.silent(duration=350) + content

    buf = io.BytesIO()
    # codec must be passed explicitly -- pydub's export() silently takes an
    # "easy_wav" shortcut (raw wave-module write, no ffmpeg invocation at all)
    # whenever format="wav" and codec is None, which skips tags entirely
    # regardless of tags= being set. pcm_s16le is WAV's own native encoding, so
    # this changes nothing about the audio -- it only forces the ffmpeg path
    # that actually writes the -metadata comment tag below.
    combined.export(buf, format="wav", codec="pcm_s16le", tags={"comment": WATERMARK_TAG})
    return buf.getvalue()


async def speak_text(text: str, profile: dict) -> dict:
    """
    Parameters
    ----------
    text    : str   Plain text to speak (caller has already stripped citations)
    profile : dict  A row from database.list_voice_profiles / get_voice_profile

    Returns
    -------
    dict {"audio": bytes, "mime": str}
    """
    base_voice = profile.get("base_tts_voice") or tts.DEFAULT_VOICE
    tts_audio, base_mime = await tts.synthesize(text, voice=base_voice)

    if profile.get("kind") == "cloned" and profile.get("status") == "ready" and profile.get("speaker_id"):
        # rvc_client.convert() is a blocking requests.post() (up to the configured
        # convert timeout, default 20s) -- offload it so one slow RVC round-trip
        # doesn't stall the event loop for every other concurrent request this
        # process is serving (this is the most-hit endpoint in the whole service).
        converted = await asyncio.to_thread(rvc_client.convert, tts_audio, profile["speaker_id"], mime=base_mime)
        rvc_used  = converted is not tts_audio
        if rvc_used:
            converted = await _add_ai_disclosure(converted, base_voice)
        return {"audio": converted, "mime": "audio/wav" if rvc_used else base_mime}

    return {"audio": tts_audio, "mime": base_mime}


def _speaker_id_for(profile_id: int, client_id: int, external_user_id: str) -> str:
    # client_id is folded in so two different client apps can't collide on the same
    # speaker_id if they happen to reuse the same external_user_id.
    return f"client{client_id}_user{external_user_id}_profile{profile_id}"


def run_training(profile_id: int, min_samples: int):
    """Background task entrypoint — see app.py POST /api/profiles/{id}/train."""
    profile = get_voice_profile(profile_id)
    if not profile:
        return

    samples = list_voice_samples(profile_id)
    if len(samples) < min_samples:
        update_voice_profile_status(
            profile_id, "failed",
            error_message=f"Cần tối thiểu {min_samples} mẫu ghi âm để huấn luyện."
        )
        return

    sample_files = []
    for s in samples:
        path = s["file_path"]
        if not os.path.exists(path):
            continue
        with open(path, "rb") as f:
            sample_files.append((os.path.basename(path), f.read()))

    if not sample_files:
        update_voice_profile_status(profile_id, "failed", error_message="Không tìm thấy file mẫu ghi âm trên đĩa.")
        return

    speaker_id = _speaker_id_for(profile_id, profile["client_id"], profile["external_user_id"])
    update_voice_profile_status(profile_id, "training", speaker_id=speaker_id, progress_message="Đang bắt đầu…")

    result = rvc_client.start_train(speaker_id, sample_files)
    if result.get("status") in ("queued", "running", "accepted"):
        _poll_until_done(profile_id, speaker_id)
        return

    # Colab unset/unreachable/errored -- train on this machine's own GPU/CPU instead
    # (see voice/rvc_local.py). Unlike the Colab path, this runs synchronously right
    # here (already inside the BackgroundTasks thread) rather than via queue + poll.
    logger.info(f"[Voice] Colab unavailable for {speaker_id} ({result.get('message')}) — training locally.")
    update_voice_profile_status(
        profile_id, "training", speaker_id=speaker_id,
        progress_message="Colab không khả dụng — đang huấn luyện trên máy chủ cục bộ…"
    )

    def _report_progress(msg: str):
        update_voice_profile_status(profile_id, "training", speaker_id=speaker_id, progress_message=msg)

    try:
        model_pth, _ = rvc_client.train_local(speaker_id, sample_files, progress_cb=_report_progress)
        update_voice_profile_status(
            profile_id, "ready", speaker_id=speaker_id, model_local_path=os.path.dirname(model_pth),
            progress_message="Hoàn tất."
        )
    except Exception as e:
        logger.error(f"[Voice] Local training failed for {speaker_id}: {e}")
        update_voice_profile_status(
            profile_id, "failed", speaker_id=speaker_id,
            error_message=f"Không thể huấn luyện (Colab và máy chủ cục bộ đều không khả dụng): {e}"
        )


def _download_and_store_model(speaker_id: str) -> str:
    """
    Best-effort local backup of the trained model into voice_storage/<speaker_id>/.
    Returns the local directory path on success, or "" if the download failed —
    the cloned voice still works via Colab's own Drive copy either way, so a
    failed backup must never block the profile from becoming 'ready'.
    """
    zip_bytes = rvc_client.download_model(speaker_id)
    if not zip_bytes:
        logger.info(f"[Voice] No local backup for {speaker_id} (Colab unreachable or model missing) — still usable via Colab.")
        return ""

    model_dir = os.path.join(VOICE_MODELS_DIR, speaker_id)
    os.makedirs(model_dir, exist_ok=True)
    try:
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            zf.extractall(model_dir)
        return model_dir
    except zipfile.BadZipFile as e:
        logger.error(f"[Voice] Corrupt model archive for {speaker_id}: {e}")
        return ""


def _poll_until_done(profile_id: int, speaker_id: str):
    waited = 0
    consecutive_network_errors = 0
    last_logged_message = None
    while waited < MAX_TRAIN_WAIT_SEC:
        time.sleep(POLL_INTERVAL_SEC)
        waited += POLL_INTERVAL_SEC

        status = rvc_client.train_status(speaker_id)
        state  = status.get("status")

        if state == "network_error":
            consecutive_network_errors += 1
            logger.info(
                f"[Voice] Poll for {speaker_id} couldn't reach Colab "
                f"({consecutive_network_errors}/{MAX_CONSECUTIVE_NETWORK_ERRORS}): {status.get('message')}"
            )
            if consecutive_network_errors >= MAX_CONSECUTIVE_NETWORK_ERRORS:
                update_voice_profile_status(
                    profile_id, "failed", speaker_id=speaker_id,
                    error_message="Mất kết nối tới Colab quá lâu trong khi huấn luyện."
                )
                return
            continue
        consecutive_network_errors = 0

        if state in ("done", "ready"):
            model_local_path = _download_and_store_model(speaker_id)
            update_voice_profile_status(
                profile_id, "ready", speaker_id=speaker_id, model_local_path=model_local_path or ""
            )
            return
        if state in ("failed", "error"):
            update_voice_profile_status(
                profile_id, "failed", speaker_id=speaker_id,
                error_message=status.get("message", "Huấn luyện thất bại trên Colab.")
            )
            return
        if state in ("queued", "running") and status.get("message"):
            # Surface Colab's own progress message (see colab/voice_server.ipynb's
            # TRAIN_JOBS store) so the frontend isn't blind while this loop polls.
            update_voice_profile_status(
                profile_id, "training", speaker_id=speaker_id, progress_message=status["message"]
            )
            # Also mirror it into the system log (unlike STT-Lab training, which logs
            # every step via engine/stt_train_engine.py, this path previously only ever
            # reached the DB's progress_message column -- invisible on the dashboard's
            # "Nhat ky he thong" panel). Gated on change since this loop polls every
            # POLL_INTERVAL_SEC and Colab often repeats the same message across polls.
            if status["message"] != last_logged_message:
                logger.info(f"[Voice] Training {speaker_id}: {status['message']}")
                last_logged_message = status["message"]
        if state == "unavailable":
            update_voice_profile_status(
                profile_id, "failed", speaker_id=speaker_id,
                error_message="Mất kết nối tới Colab trong khi huấn luyện."
            )
            return
        # queued / running — keep polling

    update_voice_profile_status(
        profile_id, "failed", speaker_id=speaker_id,
        error_message="Huấn luyện quá thời gian chờ tối đa (2 giờ)."
    )
