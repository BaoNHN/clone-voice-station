"""
engine/realism_engine.py
Manager-only "realism test": synthesizes a test clip through the SAME
pipeline real users hear (base TTS -> RVC conversion), then scores how
close it sounds to the profile owner's own uploaded samples using
resemblyzer's pretrained speaker-embedding encoder (cosine similarity
between d-vectors, the standard speaker-verification technique).

The encoder is lazy-loaded on first use, not at import time, since
resemblyzer pulls in torch and this feature isn't used in most process
lifetimes.

Decoding non-WAV original samples (.mp3/.webm/.ogg/.m4a) goes through
librosa, which needs `ffmpeg` on PATH. The synthesized test clip is always
WAV (RVC's /convert always returns WAV), so that half never needs ffmpeg.
"""

import base64
import io
import os

DEFAULT_TEST_TEXT = (
    "Xin chào, đây là đoạn văn bản thử nghiệm để kiểm tra độ giống của giọng nói "
    "nhân bản so với giọng nói gốc đã ghi âm."
)

_encoder = None


def _get_encoder():
    global _encoder
    if _encoder is None:
        from resemblyzer import VoiceEncoder
        _encoder = VoiceEncoder()
    return _encoder


def _embed_audio_bytes(audio_bytes: bytes):
    """Embeds an in-memory WAV clip (the freshly synthesized test audio)."""
    from resemblyzer import preprocess_wav
    import soundfile as sf

    wav, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32", always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    processed = preprocess_wav(wav, source_sr=sr)
    return _get_encoder().embed_utterance(processed)


def _embed_audio_file(path: str):
    """Embeds an original uploaded sample from disk (any format librosa can decode)."""
    from resemblyzer import preprocess_wav
    processed = preprocess_wav(path)
    return _get_encoder().embed_utterance(processed)


def _cosine_similarity(a, b) -> float:
    import numpy as np
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def verdict_for(score_pct: float) -> str:
    if score_pct >= 85:
        return "Rất giống"
    if score_pct >= 70:
        return "Khá giống"
    if score_pct >= 50:
        return "Hơi giống"
    return "Không giống"


async def run_realism_test(profile: dict, samples: list, text: str = "") -> dict:
    """
    Parameters
    ----------
    profile : dict  A 'cloned', 'ready' row from database.get_voice_profile()
    samples : list  database.list_voice_samples(profile_id) rows
    text    : str   Optional custom sentence to synthesize; falls back to
                     DEFAULT_TEST_TEXT

    Returns
    -------
    {
        "score_pct": float,       # similarity vs the centroid of the user's own samples
        "verdict": str,
        "sample_scores": [{"sample_id", "script_id", "score_pct"|None, "error"?}, ...],
        "synthesized_audio_b64": str,
        "synthesized_mime": "audio/wav",
        "text": str,
    }

    Raises RuntimeError (Vietnamese message, safe to show the manager directly)
    on any condition that makes scoring impossible.
    """
    import asyncio

    import numpy as np
    from voice import tts, rvc_client

    text = (text or "").strip() or DEFAULT_TEST_TEXT

    valid_samples = [s for s in samples if os.path.exists(s["file_path"])]
    if not valid_samples:
        raise RuntimeError("Không tìm thấy file mẫu ghi âm nào trên đĩa để so sánh.")

    if not profile.get("speaker_id"):
        raise RuntimeError("Giọng nói này chưa có speaker_id (chưa huấn luyện xong).")

    # Synthesize through the real playback pipeline: base TTS -> RVC convert.
    # Offloaded to a thread (like voice_engine.speak_text()) so this blocking
    # HTTP/CPU work doesn't stall concurrent requests.
    base_voice = profile.get("base_tts_voice") or tts.DEFAULT_VOICE
    tts_audio, base_mime = await tts.synthesize(text, voice=base_voice)
    synth_audio = await asyncio.to_thread(rvc_client.convert, tts_audio, profile["speaker_id"], mime=base_mime)
    if synth_audio is tts_audio:
        raise RuntimeError(
            "Không thể chuyển đổi giọng qua RVC (Colab có thể đang tắt hoặc model chưa sẵn sàng) "
            "— không có bản ghi giọng nhân bản để so sánh."
        )

    # Embed the synthesized clip + every original sample.
    synth_embed = await asyncio.to_thread(_embed_audio_bytes, synth_audio)

    sample_scores = []
    embeddings = []
    for s in valid_samples:
        try:
            emb = await asyncio.to_thread(_embed_audio_file, s["file_path"])
        except Exception as e:
            sample_scores.append({
                "sample_id": s["id"], "script_id": s["script_id"],
                "score_pct": None, "error": str(e),
            })
            continue
        embeddings.append(emb)
        pair_score = round(max(0.0, _cosine_similarity(synth_embed, emb)) * 100, 1)
        sample_scores.append({"sample_id": s["id"], "script_id": s["script_id"], "score_pct": pair_score})

    if not embeddings:
        raise RuntimeError("Không đọc được file mẫu ghi âm nào (định dạng không hỗ trợ hoặc file hỏng).")

    # Score against the CENTROID of the user's own samples, not an average of
    # per-sample scores -- matches the usual speaker-verification pattern.
    centroid = np.mean(embeddings, axis=0)
    overall_score = round(max(0.0, _cosine_similarity(synth_embed, centroid)) * 100, 1)

    return {
        "score_pct": overall_score,
        "verdict": verdict_for(overall_score),
        "sample_scores": sample_scores,
        "synthesized_audio_b64": base64.b64encode(synth_audio).decode("ascii"),
        "synthesized_mime": "audio/wav",
        "text": text,
    }
