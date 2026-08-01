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

POLL_INTERVAL_SEC = 15
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
    combined.export(buf, format="wav", tags={"comment": WATERMARK_TAG})
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
    update_voice_profile_status(profile_id, "training", speaker_id=speaker_id)

    result = rvc_client.start_train(speaker_id, sample_files)
    if result.get("status") not in ("queued", "running", "accepted"):
        update_voice_profile_status(
            profile_id, "failed", speaker_id=speaker_id,
            error_message=result.get("message", "Không thể bắt đầu huấn luyện trên Colab.")
        )
        return

    _poll_until_done(profile_id, speaker_id)


def _download_and_store_model(speaker_id: str) -> str:
    """
    Best-effort local backup of the trained model into voice_storage/<speaker_id>/.
    Returns the local directory path on success, or "" if the download failed —
    the cloned voice still works via Colab's own Drive copy either way, so a
    failed backup must never block the profile from becoming 'ready'.
    """
    zip_bytes = rvc_client.download_model(speaker_id)
    if not zip_bytes:
        print(f"[Voice] No local backup for {speaker_id} (Colab unreachable or model missing) — still usable via Colab.")
        return ""

    model_dir = os.path.join(VOICE_MODELS_DIR, speaker_id)
    os.makedirs(model_dir, exist_ok=True)
    try:
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            zf.extractall(model_dir)
        return model_dir
    except zipfile.BadZipFile as e:
        print(f"[Voice] Corrupt model archive for {speaker_id}: {e}")
        return ""


def _poll_until_done(profile_id: int, speaker_id: str):
    waited = 0
    while waited < MAX_TRAIN_WAIT_SEC:
        time.sleep(POLL_INTERVAL_SEC)
        waited += POLL_INTERVAL_SEC

        status = rvc_client.train_status(speaker_id)
        state  = status.get("status")

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
