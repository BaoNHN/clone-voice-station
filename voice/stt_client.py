"""
voice/stt_client.py
Client for STT Lab Tier 2 (LoRA fine-tune) training on Colab — mirrors
voice/rvc_client.py's shape exactly, reusing the same tunnel endpoint
(get_endpoint()) since colab/voice_server.ipynb now serves both RVC and STT
training routes off the same Flask server/cloudflared tunnel.

Unlike RVC's convert()/train_local(), this module has no local fallback of
its own — the fallback decision (Colab vs local) lives in the caller,
engine/stt_train_engine.py, same separation of concerns as
engine/voice_engine.py::run_training() does for RVC.
"""

import json

import requests

from engine.server_log import get_logger
from voice.rvc_client import get_endpoint, get_timeout

logger = get_logger()


def start_train(adapter_id: int, base_model: str, samples: list, resume_zip: bytes = None) -> dict:
    """
    Uploads training samples and asks the Colab server to enqueue an STT LoRA
    training job. Returns immediately — poll train_status() to track progress.

    samples: list of (filename, audio_bytes, reference_text) tuples.
    """
    endpoint = get_endpoint()
    if not endpoint:
        return {"status": "unavailable", "message": "Chưa cấu hình Colab endpoint."}

    try:
        files = [("files", (fname, data, "audio/wav")) for fname, data, _ in samples]
        if resume_zip:
            files.append(("resume_adapter", ("resume.zip", resume_zip, "application/zip")))
        transcripts = {fname: text for fname, _, text in samples}
        resp = requests.post(
            f"{endpoint}/stt_train",
            data={"adapter_id": adapter_id, "base_model": base_model,
                  "transcripts": json.dumps(transcripts)},
            files=files,
            timeout=60,
        )
        if resp.status_code == 200:
            return resp.json()
        return {"status": "error", "message": f"Colab trả về lỗi HTTP {resp.status_code}"}
    except requests.exceptions.RequestException as e:
        return {"status": "error", "message": f"Không kết nối được tới Colab: {e}"}


def train_status(adapter_id: int) -> dict:
    endpoint = get_endpoint()
    if not endpoint:
        return {"status": "unavailable"}

    try:
        resp = requests.get(f"{endpoint}/stt_train_status/{adapter_id}", timeout=get_timeout("short"))
        if resp.status_code == 200:
            return resp.json()
        return {"status": "error", "message": f"HTTP {resp.status_code}"}
    except requests.exceptions.RequestException as e:
        return {"status": "error", "message": str(e)}


def download_adapter(adapter_id: int) -> bytes:
    """Downloads the trained LoRA adapter (adapter_model.safetensors +
    adapter_config.json, zipped) from Colab. Returns None if unset/unreachable
    or nothing trained yet."""
    endpoint = get_endpoint()
    if not endpoint:
        return None

    try:
        resp = requests.get(f"{endpoint}/stt_train/{adapter_id}/download", timeout=get_timeout("download"))
        if resp.status_code == 200:
            return resp.content
        logger.warning(f"[STT] /stt_train/{adapter_id}/download returned {resp.status_code}")
        return None
    except requests.exceptions.RequestException as e:
        logger.warning(f"[STT] Adapter download failed for {adapter_id}: {e}")
        return None
