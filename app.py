import asyncio
import mimetypes
import os
import secrets
import uuid

import requests
from fastapi import FastAPI, Request, Form, File, UploadFile, BackgroundTasks, Header, HTTPException, Depends
from fastapi.responses import Response, HTMLResponse, JSONResponse, RedirectResponse, FileResponse
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware

from database.database import (
    init_db, get_client_by_api_key,
    get_setting, set_setting,
    has_voice_consent, record_voice_consent,
    list_voice_profiles, get_voice_profile, create_voice_profile,
    count_cloned_voice_profiles,
    rename_voice_profile, set_default_voice_profile, delete_voice_profile,
    update_voice_profile_status,
    add_voice_sample, list_voice_samples, delete_voice_sample,
    list_all_voice_profiles, list_all_voice_profiles_global,
    list_clients, create_client, delete_client, get_client, set_client_webhook,
    verify_manager_login, change_manager_password,
    create_notification, mark_notification_delivered, list_undelivered_notifications,
    VOICE_SAMPLES_DIR, MIN_TRAIN_SAMPLES, MAX_CLONED_VOICES_PER_USER,
)
from voice import rvc_client, stt
from voice.scripts import get_scripts
from engine import voice_engine, realism_engine

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.makedirs(VOICE_SAMPLES_DIR, exist_ok=True)

# Session secret for the manager dashboard login — generated once and persisted
# to disk (gitignored, same spirit as voice_station.db) rather than hardcoded,
# since this service may end up reachable by more than just localhost.
SESSION_SECRET_PATH = os.path.join(BASE_DIR, "session_secret.txt")
if not os.path.exists(SESSION_SECRET_PATH):
    with open(SESSION_SECRET_PATH, "w") as f:
        f.write(secrets.token_urlsafe(32))
with open(SESSION_SECRET_PATH) as f:
    SESSION_SECRET = f.read().strip()

app = FastAPI()
app.add_middleware(SessionMiddleware, secret_key=SESSION_SECRET, max_age=7200)
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

init_db()


# ── Auth: client apps (API key, per-request) ────────────────────────────────────
def require_client(x_api_key: str = Header(None)) -> dict:
    client = get_client_by_api_key(x_api_key)
    if not client:
        raise HTTPException(status_code=401, detail="Invalid or missing X-Api-Key")
    return client


# ── Auth: manager dashboard (session, entirely separate from client apps) ──────
def manager_logged_in(request: Request) -> bool:
    return bool(request.session.get("manager"))


def require_manager(request: Request):
    if not manager_logged_in(request):
        raise HTTPException(status_code=401, detail="Not logged in")
    return request.session["manager"]


def _owns_cloned(profile: dict, client_id: int, external_user_id: str) -> bool:
    return bool(
        profile and profile["kind"] == "cloned"
        and profile["client_id"] == client_id
        and profile["external_user_id"] == external_user_id
    )


# ── Notifications: manager-triggered delete/disable → end user ────────────────
def _notify_profile_event(profile: dict, event: str, message: str):
    """Records the event and best-effort delivers it to the owning client
    app's registered webhook. Never raises — a client being unreachable must
    not block the manager action (delete/disable) that triggered this."""
    client_id        = profile.get("client_id")
    external_user_id = profile.get("external_user_id")
    if not client_id or not external_user_id:
        return  # builtin voices have no owning end user to notify

    notification_id = create_notification(
        client_id, external_user_id, profile["id"], profile["name"], event, message
    )

    client = get_client(client_id)
    webhook_url = client.get("webhook_url") if client else None
    if not webhook_url:
        return  # no webhook registered — client picks this up via GET /api/notifications instead

    try:
        resp = requests.post(
            webhook_url,
            headers={"X-Api-Key": client["api_key"]},
            json={
                "notification_id": notification_id,
                "event": event,
                "external_user_id": external_user_id,
                "profile_id": profile["id"],
                "profile_name": profile["name"],
                "message": message,
            },
            timeout=8,
        )
        if resp.status_code == 200:
            mark_notification_delivered(notification_id)
    except requests.exceptions.RequestException:
        pass  # left undelivered — the client's polling fallback will pick it up


@app.get("/api/health")
async def health_route():
    return {"status": "ok"}


# ── Manager dashboard: login/logout/pages ───────────────────────────────────────
@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    if manager_logged_in(request):
        return RedirectResponse("/", status_code=302)
    return templates.TemplateResponse(request, "login.html", {"error": None})


@app.post("/login", response_class=HTMLResponse)
async def login_submit(request: Request, username: str = Form(...), password: str = Form(...)):
    if not verify_manager_login(username, password):
        return templates.TemplateResponse(
            request, "login.html",
            {"error": "Sai tên đăng nhập hoặc mật khẩu."}, status_code=401,
        )
    request.session["manager"] = username
    return RedirectResponse("/", status_code=302)


@app.post("/logout")
async def logout_route(request: Request):
    request.session.clear()
    return RedirectResponse("/login", status_code=302)


@app.get("/", response_class=HTMLResponse)
async def dashboard_page(request: Request):
    if not manager_logged_in(request):
        return RedirectResponse("/login", status_code=302)
    return templates.TemplateResponse(request, "dashboard.html", {
        "manager": request.session["manager"],
        "min_samples": MIN_TRAIN_SAMPLES,
    })


# ── Manager dashboard: RVC connection + tuning ──────────────────────────────────
@app.get("/manager/rvc_config")
async def get_rvc_config_route(manager: str = Depends(require_manager)):
    return {
        "endpoint":    get_setting("rvc_endpoint"),
        "available":   rvc_client.is_available(),
        "pitch":       rvc_client.get_pitch(),
        "index_rate":  rvc_client.get_index_rate(),
        "timeout_convert":    rvc_client.get_timeout("convert"),
        "timeout_short":      rvc_client.get_timeout("short"),
        "timeout_download":   rvc_client.get_timeout("download"),
        "timeout_f5tts":      rvc_client.get_timeout("f5tts"),
        "timeout_transcribe": rvc_client.get_timeout("transcribe"),
    }


@app.post("/manager/rvc_config")
async def set_rvc_config_route(request: Request, manager: str = Depends(require_manager)):
    data = await request.json()

    if "endpoint" in data:
        set_setting("rvc_endpoint", (data.get("endpoint") or "").strip())
    if "pitch" in data:
        set_setting("rvc_pitch", str(int(data["pitch"])))
    if "index_rate" in data:
        set_setting("rvc_index_rate", str(float(data["index_rate"])))
    for key in ("convert", "short", "download", "f5tts", "transcribe"):
        if f"timeout_{key}" in data:
            set_setting(f"rvc_timeout_{key}", str(int(data[f"timeout_{key}"])))

    return {"status": "ok", "available": rvc_client.is_available()}


# ── Manager dashboard: client apps ──────────────────────────────────────────────
@app.get("/manager/clients")
async def list_clients_route(manager: str = Depends(require_manager)):
    return list_clients()


@app.post("/manager/clients")
async def create_client_route(request: Request, manager: str = Depends(require_manager)):
    data = await request.json()
    name = (data.get("name") or "").strip()
    if not name:
        raise HTTPException(status_code=400, detail="Vui lòng nhập tên client.")
    try:
        return create_client(name)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.delete("/manager/clients/{client_id}")
async def delete_client_route(client_id: int, manager: str = Depends(require_manager)):
    try:
        delete_client(client_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"status": "ok"}


@app.post("/manager/clients/{client_id}/webhook")
async def set_client_webhook_route(client_id: int, request: Request, manager: str = Depends(require_manager)):
    """Manual override — a client app can also self-register via POST
    /api/webhook using its own API key. This lets a manager set/clear it
    directly if that integration step hasn't happened yet."""
    if not get_client(client_id):
        raise HTTPException(status_code=404, detail="Không tìm thấy client.")
    data = await request.json()
    set_client_webhook(client_id, (data.get("webhook_url") or "").strip())
    return {"status": "ok"}


# ── Manager dashboard: every voice profile, across every client ────────────────
@app.get("/manager/profiles")
async def list_all_profiles_route(manager: str = Depends(require_manager)):
    return list_all_voice_profiles_global()


@app.post("/manager/profiles/{profile_id}/retrain")
async def manager_retrain_route(profile_id: int, background_tasks: BackgroundTasks,
                                 manager: str = Depends(require_manager)):
    profile = get_voice_profile(profile_id)
    if not profile or profile["kind"] != "cloned":
        raise HTTPException(status_code=404, detail="Không tìm thấy giọng nói.")

    samples = list_voice_samples(profile_id)
    if len(samples) < MIN_TRAIN_SAMPLES:
        raise HTTPException(status_code=400, detail=f"Cần tối thiểu {MIN_TRAIN_SAMPLES} mẫu ghi âm.")

    update_voice_profile_status(profile_id, "training")
    background_tasks.add_task(voice_engine.run_training, profile_id, MIN_TRAIN_SAMPLES)
    return {"status": "ok"}


async def _optional_reason(request: Request) -> str:
    try:
        body = await request.json()
        return (body.get("reason") or "").strip()
    except Exception:
        return ""


@app.post("/manager/profiles/{profile_id}/disable")
async def manager_disable_route(profile_id: int, request: Request, manager: str = Depends(require_manager)):
    profile = get_voice_profile(profile_id)
    if not profile or profile["kind"] != "cloned":
        raise HTTPException(status_code=404, detail="Không tìm thấy giọng nói.")
    reason = await _optional_reason(request)

    update_voice_profile_status(profile_id, "failed", error_message="Đã bị quản trị viên vô hiệu hóa.")

    message = f'Giọng nói "{profile["name"]}" của bạn đã bị quản trị viên vô hiệu hóa.'
    if reason:
        message += f" Lý do: {reason}"
    _notify_profile_event(profile, "voice_profile_disabled", message)

    return {"status": "ok"}


@app.delete("/manager/profiles/{profile_id}")
async def manager_delete_profile_route(profile_id: int, request: Request, manager: str = Depends(require_manager)):
    profile = get_voice_profile(profile_id)
    if not profile or profile["kind"] != "cloned":
        raise HTTPException(status_code=404, detail="Không tìm thấy giọng nói.")
    reason = await _optional_reason(request)

    remote_cleaned = True
    if profile.get("speaker_id"):
        result = rvc_client.delete_model(profile["speaker_id"])
        remote_cleaned = result.get("status") == "ok"

    message = f'Giọng nói "{profile["name"]}" của bạn đã bị quản trị viên xoá.'
    if reason:
        message += f" Lý do: {reason}"
    _notify_profile_event(profile, "voice_profile_deleted", message)

    delete_voice_profile(profile_id)
    return {"status": "ok", "remote_cleaned": remote_cleaned}


# ── Manager dashboard: voice realism test ───────────────────────────────────────
# Lets the manager judge how close a trained clone actually sounds to the end
# user's own recordings — synthesizes a test clip through the real playback
# pipeline (base TTS -> RVC) and scores it against the profile's uploaded
# samples via resemblyzer speaker-embedding cosine similarity (see
# engine/realism_engine.py). Manager-only: this plays back an end user's raw
# voice recordings, not just metadata about them.
@app.get("/manager/profiles/{profile_id}/samples")
async def manager_list_samples_route(profile_id: int, manager: str = Depends(require_manager)):
    profile = get_voice_profile(profile_id)
    if not profile or profile["kind"] != "cloned":
        raise HTTPException(status_code=404, detail="Không tìm thấy giọng nói.")
    return list_voice_samples(profile_id)


@app.get("/manager/profiles/{profile_id}/samples/{sample_id}/audio")
async def manager_sample_audio_route(profile_id: int, sample_id: int, manager: str = Depends(require_manager)):
    samples = list_voice_samples(profile_id)
    sample  = next((s for s in samples if s["id"] == sample_id), None)
    if not sample or not os.path.exists(sample["file_path"]):
        raise HTTPException(status_code=404, detail="Không tìm thấy file mẫu ghi âm.")
    mime = mimetypes.guess_type(sample["file_path"])[0] or "application/octet-stream"
    return FileResponse(sample["file_path"], media_type=mime)


@app.post("/manager/profiles/{profile_id}/realism_test")
async def manager_realism_test_route(profile_id: int, request: Request, manager: str = Depends(require_manager)):
    profile = get_voice_profile(profile_id)
    if not profile or profile["kind"] != "cloned":
        raise HTTPException(status_code=404, detail="Không tìm thấy giọng nói.")
    if profile["status"] != "ready":
        raise HTTPException(status_code=400, detail="Giọng nói này chưa ở trạng thái sẵn sàng.")

    try:
        body = await request.json()
    except Exception:
        body = {}
    text = (body.get("text") or "").strip()

    samples = list_voice_samples(profile_id)
    try:
        return await realism_engine.run_realism_test(profile, samples, text=text)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/manager/change_password")
async def manager_change_password_route(request: Request, manager: str = Depends(require_manager)):
    data         = await request.json()
    new_password = data.get("new_password") or ""
    if len(new_password) < 8:
        raise HTTPException(status_code=400, detail="Mật khẩu mới cần tối thiểu 8 ký tự.")
    change_manager_password(manager, new_password)
    return {"status": "ok"}


# ── Scripts + profiles (any authenticated client, scoped to its end user) ──────
@app.get("/api/scripts")
async def scripts_route(client: dict = Depends(require_client)):
    return get_scripts()


@app.get("/api/consent")
async def get_consent_route(external_user_id: str, client: dict = Depends(require_client)):
    return {"consent": has_voice_consent(client["id"], external_user_id)}


@app.post("/api/consent")
async def post_consent_route(request: Request, client: dict = Depends(require_client)):
    data             = await request.json()
    external_user_id = str(data.get("external_user_id") or "")
    if not external_user_id:
        raise HTTPException(status_code=400, detail="external_user_id required")
    record_voice_consent(client["id"], external_user_id)
    return {"status": "ok"}


@app.get("/api/profiles")
async def list_profiles_route(external_user_id: str, client: dict = Depends(require_client)):
    return list_voice_profiles(client["id"], external_user_id)


@app.post("/api/profiles")
async def create_profile_route(request: Request, client: dict = Depends(require_client)):
    data             = await request.json()
    external_user_id = str(data.get("external_user_id") or "")
    name             = (data.get("name") or "").strip()
    if not external_user_id or not name:
        raise HTTPException(status_code=400, detail="external_user_id and name required")

    if not has_voice_consent(client["id"], external_user_id):
        raise HTTPException(status_code=403, detail="Vui lòng đồng ý với điều khoản sử dụng giọng nói trước.")

    if count_cloned_voice_profiles(client["id"], external_user_id) >= MAX_CLONED_VOICES_PER_USER:
        raise HTTPException(
            status_code=400,
            detail=f"Đã có tối đa {MAX_CLONED_VOICES_PER_USER} giọng nói riêng. Vui lòng xoá một giọng nói cũ trước."
        )

    profile_id = create_voice_profile(client["id"], external_user_id, name)
    return {"status": "ok", "profile_id": profile_id}


@app.put("/api/profiles/{profile_id}")
async def update_profile_route(profile_id: int, request: Request, client: dict = Depends(require_client)):
    profile = get_voice_profile(profile_id)
    if not profile:
        raise HTTPException(status_code=404, detail="Không tìm thấy giọng nói.")

    data             = await request.json()
    external_user_id = str(data.get("external_user_id") or "")
    is_own_cloned    = _owns_cloned(profile, client["id"], external_user_id)
    is_selectable    = profile["kind"] == "builtin" or is_own_cloned
    if not is_selectable:
        raise HTTPException(status_code=403, detail="Unauthorized")

    if "name" in data and is_own_cloned:
        name = (data.get("name") or "").strip()
        if name:
            rename_voice_profile(profile_id, name)
    if data.get("is_default"):
        set_default_voice_profile(client["id"], external_user_id, profile_id)

    return {"status": "ok"}


@app.delete("/api/profiles/{profile_id}")
async def delete_profile_route(profile_id: int, external_user_id: str, client: dict = Depends(require_client)):
    profile = get_voice_profile(profile_id)
    if not _owns_cloned(profile, client["id"], external_user_id):
        raise HTTPException(status_code=404, detail="Không tìm thấy giọng nói của bạn.")

    remote_cleaned = True
    if profile.get("speaker_id"):
        result = rvc_client.delete_model(profile["speaker_id"])
        remote_cleaned = result.get("status") == "ok"

    delete_voice_profile(profile_id)
    return {
        "status": "ok",
        "message": None if remote_cleaned else
            "Đã xoá trong ứng dụng, nhưng không xoá được model trên Colab (Colab có thể đang tắt) — cần dọn thủ công sau."
    }


@app.post("/api/profiles/{profile_id}/samples")
async def upload_sample_route(
    profile_id: int,
    external_user_id: str = Form(...),
    script_id: str = Form(...),
    audio: UploadFile = File(...),
    client: dict = Depends(require_client),
):
    profile = get_voice_profile(profile_id)
    if not _owns_cloned(profile, client["id"], external_user_id):
        raise HTTPException(status_code=404, detail="Không tìm thấy giọng nói của bạn.")
    if not has_voice_consent(client["id"], external_user_id):
        raise HTTPException(status_code=403, detail="Vui lòng đồng ý với điều khoản sử dụng giọng nói trước.")

    sample_dir = os.path.join(VOICE_SAMPLES_DIR, str(profile_id))
    os.makedirs(sample_dir, exist_ok=True)

    ext = ".wav"
    fname_lower = (audio.filename or "").lower()
    for candidate in (".mp3", ".webm", ".ogg", ".m4a", ".wav"):
        if fname_lower.endswith(candidate):
            ext = candidate
            break

    file_path = os.path.join(sample_dir, f"{uuid.uuid4()}{ext}")
    content   = await audio.read()
    with open(file_path, "wb") as f:
        f.write(content)

    sample_id = add_voice_sample(profile_id, script_id, file_path)
    samples   = list_voice_samples(profile_id)
    return {"status": "ok", "sample_id": sample_id, "sample_count": len(samples)}


@app.get("/api/profiles/{profile_id}/samples")
async def list_samples_route(profile_id: int, external_user_id: str, client: dict = Depends(require_client)):
    profile = get_voice_profile(profile_id)
    if not _owns_cloned(profile, client["id"], external_user_id):
        raise HTTPException(status_code=404, detail="Không tìm thấy giọng nói của bạn.")
    return list_voice_samples(profile_id)


@app.delete("/api/profiles/{profile_id}/samples/{sample_id}")
async def delete_sample_route(profile_id: int, sample_id: int, external_user_id: str, client: dict = Depends(require_client)):
    profile = get_voice_profile(profile_id)
    if not _owns_cloned(profile, client["id"], external_user_id):
        raise HTTPException(status_code=404, detail="Không tìm thấy giọng nói của bạn.")
    delete_voice_sample(sample_id)
    return {"status": "ok"}


@app.post("/api/profiles/{profile_id}/train")
async def train_profile_route(profile_id: int, request: Request, background_tasks: BackgroundTasks,
                               client: dict = Depends(require_client)):
    data             = await request.json()
    external_user_id = str(data.get("external_user_id") or "")
    profile          = get_voice_profile(profile_id)
    if not _owns_cloned(profile, client["id"], external_user_id):
        raise HTTPException(status_code=404, detail="Không tìm thấy giọng nói của bạn.")
    if profile["status"] == "training":
        raise HTTPException(status_code=400, detail="Giọng nói này đang được huấn luyện.")

    samples = list_voice_samples(profile_id)
    if len(samples) < MIN_TRAIN_SAMPLES:
        raise HTTPException(
            status_code=400,
            detail=f"Cần ghi âm tối thiểu {MIN_TRAIN_SAMPLES} đoạn văn bản (hiện có {len(samples)})."
        )

    update_voice_profile_status(profile_id, "training")
    background_tasks.add_task(voice_engine.run_training, profile_id, MIN_TRAIN_SAMPLES)
    return {"status": "ok", "message": "Đã gửi yêu cầu huấn luyện tới Colab."}


@app.get("/api/profiles/{profile_id}/status")
async def profile_status_route(profile_id: int, external_user_id: str, client: dict = Depends(require_client)):
    profile = get_voice_profile(profile_id)
    if not profile or (profile["kind"] == "cloned" and not _owns_cloned(profile, client["id"], external_user_id)):
        raise HTTPException(status_code=404, detail="Không tìm thấy.")
    return profile


@app.post("/api/transcribe")
async def transcribe_route(
    audio: UploadFile = File(...),
    language: str = Form(None),
    client: dict = Depends(require_client),
):
    """Speech-to-Text — input half of the voice loop (see /api/speak below for
    the output half). A client app posts recorded microphone audio here and
    gets back plain text to feed into its own assistant as a normal text
    query; this service never sees or needs to know what that query means.

    Tries the Colab-hosted PhoWhisper endpoint first (Vietnamese-tuned, see
    voice/rvc_client.py's transcribe_remote() and colab/voice_server.ipynb) and
    falls back to the local, CPU-only openai-whisper model (voice/stt.py)
    whenever Colab is unset/unreachable — same degrade-gracefully contract as
    /api/speak's RVC conversion."""
    audio_bytes = await audio.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Không có dữ liệu âm thanh.")

    mime = audio.content_type or mimetypes.guess_type(audio.filename or "")[0] or "audio/webm"

    result = await asyncio.to_thread(rvc_client.transcribe_remote, audio_bytes, mime, language or None)
    if result is None:
        try:
            result = await asyncio.to_thread(stt.transcribe, audio_bytes, mime, language or None)
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"Không thể chuyển giọng nói thành văn bản: {e}")

    if not result.get("text"):
        raise HTTPException(status_code=422, detail="Không nhận diện được nội dung giọng nói.")

    return result


@app.post("/api/speak")
async def speak_route(request: Request, client: dict = Depends(require_client)):
    data             = await request.json()
    text             = (data.get("text") or "").strip()
    external_user_id = str(data.get("external_user_id") or "")
    profile_id       = data.get("profile_id")
    try:
        profile_id = int(profile_id) if profile_id not in (None, "") else None
    except (TypeError, ValueError):
        profile_id = None

    if not text:
        raise HTTPException(status_code=400, detail="Không có nội dung để đọc.")

    profile = get_voice_profile(profile_id) if profile_id else None
    if profile and profile["kind"] == "cloned" and not _owns_cloned(profile, client["id"], external_user_id):
        raise HTTPException(status_code=403, detail="Unauthorized")
    if not profile:
        profiles = list_voice_profiles(client["id"], external_user_id)
        profile  = next((p for p in profiles if p["kind"] == "builtin"), None)

    if not profile:
        raise HTTPException(status_code=404, detail="Không tìm thấy giọng nói khả dụng.")

    result = await voice_engine.speak_text(text, profile)
    return Response(content=result["audio"], media_type=result["mime"])


# ── Admin (scoped to the calling client's own end users) ────────────────────────
@app.get("/api/admin/voice_models")
async def list_all_models_route(client: dict = Depends(require_client)):
    return list_all_voice_profiles(client["id"])


@app.post("/api/admin/voice_models/{profile_id}/retrain")
async def admin_retrain_route(profile_id: int, background_tasks: BackgroundTasks,
                               client: dict = Depends(require_client)):
    profile = get_voice_profile(profile_id)
    if not profile or profile["kind"] != "cloned" or profile["client_id"] != client["id"]:
        raise HTTPException(status_code=404, detail="Không tìm thấy giọng nói.")

    samples = list_voice_samples(profile_id)
    if len(samples) < MIN_TRAIN_SAMPLES:
        raise HTTPException(status_code=400, detail=f"Cần tối thiểu {MIN_TRAIN_SAMPLES} mẫu ghi âm.")

    update_voice_profile_status(profile_id, "training")
    background_tasks.add_task(voice_engine.run_training, profile_id, MIN_TRAIN_SAMPLES)
    return {"status": "ok"}


@app.post("/api/admin/voice_models/{profile_id}/disable")
async def admin_disable_route(profile_id: int, client: dict = Depends(require_client)):
    profile = get_voice_profile(profile_id)
    if not profile or profile["kind"] != "cloned" or profile["client_id"] != client["id"]:
        raise HTTPException(status_code=404, detail="Không tìm thấy giọng nói.")
    update_voice_profile_status(profile_id, "failed", error_message="Đã bị quản trị viên vô hiệu hóa.")
    return {"status": "ok"}


@app.delete("/api/admin/voice_models/{profile_id}")
async def admin_delete_route(profile_id: int, client: dict = Depends(require_client)):
    profile = get_voice_profile(profile_id)
    if not profile or profile["kind"] != "cloned" or profile["client_id"] != client["id"]:
        raise HTTPException(status_code=404, detail="Không tìm thấy giọng nói.")

    remote_cleaned = True
    if profile.get("speaker_id"):
        result = rvc_client.delete_model(profile["speaker_id"])
        remote_cleaned = result.get("status") == "ok"

    delete_voice_profile(profile_id)
    return {
        "status": "ok",
        "message": None if remote_cleaned else
            "Đã xoá trong ứng dụng, nhưng không xoá được model trên Colab (Colab có thể đang tắt) — cần dọn thủ công sau."
    }


@app.get("/api/rvc_endpoint")
async def get_rvc_endpoint_route(client: dict = Depends(require_client)):
    return {"endpoint": get_setting("rvc_endpoint"), "available": rvc_client.is_available()}


@app.post("/api/rvc_endpoint")
async def set_rvc_endpoint_route(request: Request, client: dict = Depends(require_client)):
    data     = await request.json()
    endpoint = (data.get("endpoint") or "").strip()
    set_setting("rvc_endpoint", endpoint)
    return {"status": "ok", "available": rvc_client.is_available()}


# ── Notifications: client self-registration + polling fallback ─────────────────
# When a manager deletes/disables a voice (see /manager/profiles/*), the end
# user needs to find out. Delivery is webhook-first (POST to the URL below,
# see _notify_profile_event), with polling as the fallback for whenever the
# webhook is unset or was unreachable at the time.
@app.post("/api/webhook")
async def set_webhook_route(request: Request, client: dict = Depends(require_client)):
    """A client app registers/updates its own callback URL. clone-voice-station
    POSTs {notification_id, event, external_user_id, profile_id, profile_name,
    message} to it (with X-Api-Key: <this client's key>) whenever a manager
    deletes/disables one of its end users' voices."""
    data        = await request.json()
    webhook_url = (data.get("webhook_url") or "").strip()
    set_client_webhook(client["id"], webhook_url)
    return {"status": "ok"}


@app.get("/api/notifications")
async def list_notifications_route(external_user_id: str, client: dict = Depends(require_client)):
    """Polling fallback — returns notifications not yet marked delivered for
    this end user. Call POST /api/notifications/{id}/ack after processing each
    one so it isn't returned again."""
    return list_undelivered_notifications(client["id"], external_user_id)


@app.post("/api/notifications/{notification_id}/ack")
async def ack_notification_route(notification_id: int, client: dict = Depends(require_client)):
    mark_notification_delivered(notification_id, client_id=client["id"])
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8090, reload=False)
