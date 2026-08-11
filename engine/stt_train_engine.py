"""
engine/stt_train_engine.py
Orchestrates STT Lab Tier 2 (LoRA fine-tune) training — the STT counterpart to
engine/voice_engine.py::run_training(), called via FastAPI BackgroundTasks from
POST /api/stt/adapters/{id}/train.

backend selection:
  "colab" — tries voice/stt_client.py against the Colab tunnel; fails the
            adapter explicitly (no fallback) if Colab is unset/unreachable,
            since forcing a backend is how a guest *compares* the two.
  "local" — always runs on this machine, via the single-worker queue below.
  "auto"  — tries Colab first, falls back to local on failure. Same semantics
            as RVC's run_training() today.

Unlike RVC's local-fallback training (voice/rvc_local.py, invoked straight
from whichever BackgroundTasks thread called it, with zero cross-job
concurrency guard), STT Lab has no API-key gate at all — a guest walks
straight in and registers. Local training jobs are therefore never run
inline: they're handed to a single dedicated worker thread + queue.Queue(),
directly mirroring colab/voice_server.ipynb's own "one GPU -> one job at a
time" pattern, just moved server-side for the local-fallback path.
"""

import os
import queue
import shutil
import threading
import time
import zipfile
from io import BytesIO

from database.database import (
    MIN_STT_TRAIN_SAMPLES, STT_MODELS_DIR, get_stt_adapter, list_stt_training_samples,
    update_stt_adapter_training,
)
from engine.server_log import get_logger
from voice import stt_client, stt_local_train

logger = get_logger()

POLL_INTERVAL_SEC = 15
MAX_TRAIN_WAIT_SEC = 2 * 60 * 60  # 2 hours, same cap as RVC training

os.makedirs(STT_MODELS_DIR, exist_ok=True)


def _adapter_output_dir(adapter_id: int) -> str:
    return os.path.join(STT_MODELS_DIR, str(adapter_id))


def _load_samples(adapter_id: int) -> list:
    samples = []
    for s in list_stt_training_samples(adapter_id):
        if os.path.exists(s["audio_path"]):
            samples.append(s)
    return samples


def _zip_dir(path: str) -> bytes:
    buf = BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for fname in os.listdir(path):
            zf.write(os.path.join(path, fname), arcname=fname)
    return buf.getvalue()


# ── Local training queue (one GPU -> one job at a time) ─────────────────────
_local_queue = queue.Queue()


def _run_local(adapter_id: int, samples: list, base_model: str, resume_from_path: str, holdout_samples: list = None):
    output_dir = _adapter_output_dir(adapter_id)

    def _report_progress(msg: str):
        update_stt_adapter_training(adapter_id, "training", progress_message=msg)

    try:
        stt_local_train.train(
            adapter_id, samples, base_model, output_dir,
            resume_from_path=resume_from_path, progress_cb=_report_progress,
            holdout_samples=holdout_samples,
        )
        update_stt_adapter_training(
            adapter_id, "ready", adapter_path=output_dir, backend_used="local",
            progress_message="Hoàn tất.",
        )
    except Exception as e:
        logger.error(f"[STT-train] Local training failed for adapter {adapter_id}: {e}")
        shutil.rmtree(output_dir, ignore_errors=True)
        update_stt_adapter_training(adapter_id, "failed", error_message=str(e))


def _local_worker_loop():
    while True:
        adapter_id, samples, base_model, resume_from_path, holdout_samples = _local_queue.get()
        try:
            _run_local(adapter_id, samples, base_model, resume_from_path, holdout_samples)
        finally:
            _local_queue.task_done()


threading.Thread(target=_local_worker_loop, daemon=True).start()


def _enqueue_local(adapter_id: int, samples: list, base_model: str, resume_from_path: str, note: str,
                    holdout_samples: list = None):
    update_stt_adapter_training(adapter_id, "training", progress_message=note)
    _local_queue.put((adapter_id, samples, base_model, resume_from_path, holdout_samples))


# ── Colab polling ─────────────────────────────────────────────────────────
def _poll_colab_until_done(adapter_id: int):
    waited = 0
    while waited < MAX_TRAIN_WAIT_SEC:
        time.sleep(POLL_INTERVAL_SEC)
        waited += POLL_INTERVAL_SEC

        status = stt_client.train_status(adapter_id)
        state = status.get("status")

        if state in ("done", "ready"):
            zip_bytes = stt_client.download_adapter(adapter_id)
            output_dir = _adapter_output_dir(adapter_id)
            if zip_bytes:
                os.makedirs(output_dir, exist_ok=True)
                with zipfile.ZipFile(BytesIO(zip_bytes)) as zf:
                    zf.extractall(output_dir)
                update_stt_adapter_training(
                    adapter_id, "ready", adapter_path=output_dir, backend_used="colab",
                    progress_message="Hoàn tất.",
                )
            else:
                update_stt_adapter_training(
                    adapter_id, "failed",
                    error_message="Huấn luyện xong trên Colab nhưng không tải được adapter về.",
                )
            return
        if state in ("failed", "error"):
            update_stt_adapter_training(
                adapter_id, "failed", error_message=status.get("message", "Huấn luyện thất bại trên Colab.")
            )
            return
        if state in ("queued", "running") and status.get("message"):
            update_stt_adapter_training(adapter_id, "training", progress_message=status["message"])
        if state == "unavailable":
            update_stt_adapter_training(adapter_id, "failed", error_message="Mất kết nối tới Colab trong khi huấn luyện.")
            return
        # queued / running — keep polling

    update_stt_adapter_training(adapter_id, "failed", error_message="Huấn luyện quá thời gian chờ tối đa (2 giờ).")


def run_training(adapter_id: int, backend: str = "auto"):
    """Background task entrypoint — see app.py POST /api/stt/adapters/{id}/train."""
    adapter = get_stt_adapter(adapter_id)
    if not adapter:
        return

    all_samples = _load_samples(adapter_id)
    # is_holdout samples (a genuinely independent test set, e.g. a HuggingFace dataset's
    # own official "test" split -- see voice/stt_local_train.py's train() docstring) are
    # never part of the trainable pool, only the final gate.
    samples = [s for s in all_samples if not s.get("is_holdout")]
    holdout_samples = [s for s in all_samples if s.get("is_holdout")]
    if len(samples) < MIN_STT_TRAIN_SAMPLES:
        update_stt_adapter_training(
            adapter_id, "failed",
            error_message=f"Cần tối thiểu {MIN_STT_TRAIN_SAMPLES} mẫu (audio, transcript) để huấn luyện.",
        )
        return

    base_model = adapter["base_model"]
    resume_from_path = adapter.get("resume_from_path")
    update_stt_adapter_training(adapter_id, "training", progress_message="Đang bắt đầu…")

    if backend == "local":
        _enqueue_local(adapter_id, samples, base_model, resume_from_path, "Đang chờ trong hàng đợi cục bộ…",
                        holdout_samples=holdout_samples)
        return

    # backend in ("colab", "auto") -- try Colab. Colab's own training pipeline (see
    # colab/voice_server.ipynb) doesn't yet support a holdout-based gate -- out of scope
    # here -- so holdout samples are excluded from what gets uploaded rather than silently
    # trained on (which would defeat their whole purpose).
    colab_samples = []
    for s in samples:
        with open(s["audio_path"], "rb") as f:
            colab_samples.append((os.path.basename(s["audio_path"]), f.read(), s["reference_text"]))
    resume_zip = _zip_dir(resume_from_path) if resume_from_path and os.path.isdir(resume_from_path) else None
    result = stt_client.start_train(adapter_id, base_model, colab_samples, resume_zip=resume_zip)

    if result.get("status") in ("queued", "running", "accepted"):
        _poll_colab_until_done(adapter_id)
        return

    if backend == "colab":
        # Explicitly forced Colab -- no silent fallback, that's the whole point
        # of letting a guest pick a backend to compare them.
        update_stt_adapter_training(
            adapter_id, "failed",
            error_message=f"Colab không khả dụng: {result.get('message', 'không rõ lý do')}",
        )
        return

    # backend == "auto" and Colab unavailable -- fall back to local, same
    # semantics as RVC's run_training() today.
    logger.info(f"[STT-train] Colab unavailable for adapter {adapter_id} ({result.get('message')}) — falling back to local.")
    _enqueue_local(
        adapter_id, samples, base_model, resume_from_path,
        "Colab không khả dụng — đang xếp hàng huấn luyện cục bộ…",
        holdout_samples=holdout_samples,
    )
