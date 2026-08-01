"""
engine/realism_engine.py
Manager-only "realism test" (see app.py POST /manager/profiles/{id}/realism_test):
synthesizes a test clip through the SAME pipeline real users hear (base TTS ->
RVC conversion, see voice_engine.speak_text), then scores how close that clip
sounds to the profile owner's own uploaded samples using resemblyzer's
pretrained speaker-embedding encoder — the standard speaker-verification
technique (cosine similarity between d-vectors), not a subjective listen.

resemblyzer pulls in torch — a real dependency this service didn't previously
have — so the encoder is lazy-loaded on first use rather than at import time,
keeping normal app.py startup fast for the (common) case this feature is
never used in a given process lifetime.

Decoding non-WAV original samples (.mp3/.webm/.ogg/.m4a — see app.py's
upload_sample_route) goes through librosa, which needs an `ffmpeg` binary on
PATH for anything other than .wav. The synthesized test clip itself is always
WAV (RVC's /convert always returns WAV, see rvc_client.convert), so that half
never needs ffmpeg.
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
    import numpy as np
    from voice import tts, rvc_client

    text = (text or "").strip() or DEFAULT_TEST_TEXT

    valid_samples = [s for s in samples if os.path.exists(s["file_path"])]
    if not valid_samples:
        raise RuntimeError("Không tìm thấy file mẫu ghi âm nào trên đĩa để so sánh.")

    if not profile.get("speaker_id"):
        raise RuntimeError("Giọng nói này chưa có speaker_id (chưa huấn luyện xong).")

    # 1. Synthesize through the real playback pipeline: base TTS -> RVC convert.
    base_voice = profile.get("base_tts_voice") or tts.DEFAULT_VOICE
    tts_audio, base_mime = await tts.synthesize(text, voice=base_voice)
    synth_audio = rvc_client.convert(tts_audio, profile["speaker_id"], mime=base_mime)
    if synth_audio is tts_audio:
        raise RuntimeError(
            "Không thể chuyển đổi giọng qua RVC (Colab có thể đang tắt hoặc model chưa sẵn sàng) "
            "— không có bản ghi giọng nhân bản để so sánh."
        )

    # 2. Embed the synthesized clip + every original sample.
    synth_embed = _embed_audio_bytes(synth_audio)

    sample_scores = []
    embeddings = []
    for s in valid_samples:
        try:
            emb = _embed_audio_file(s["file_path"])
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

    # Overall score = similarity to the CENTROID of the user's own samples —
    # more stable than averaging per-sample pair scores, and matches the usual
    # speaker-verification pattern of comparing an utterance against a
    # speaker's full enrolled embedding rather than one reference clip at a time.
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
