import asyncio
import io
import json
import mimetypes
import os
import re
import secrets
import shutil
import sqlite3
import time
import uuid
import zipfile

import soundfile as sf

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
    create_stt_guest, verify_stt_guest_login, get_stt_guest,
    list_stt_adapters, get_stt_adapter, create_stt_adapter,
    rename_stt_adapter, update_stt_adapter_hotwords, delete_stt_adapter,
    delete_stt_guest_account,
    MAX_HOTWORDS_PER_ADAPTER, MAX_HOTWORD_LEN,
    count_stt_adapters, MAX_STT_ADAPTERS_PER_GUEST, ALLOWED_STT_BASE_MODELS,
    MIN_STT_TRAIN_SAMPLES, MAX_STT_TRAIN_SAMPLES, MAX_STT_SAMPLE_DURATION_SEC,
    STT_SAMPLES_DIR, STT_MODELS_DIR,
    add_stt_training_sample, list_stt_training_samples, get_stt_training_sample,
    update_stt_training_sample_text,
    delete_stt_training_sample, set_stt_adapter_resume_path,
    list_all_stt_adapters_global, publish_stt_adapter, unpublish_stt_adapter,
    get_published_stt_adapter_for_client,
    set_default_stt_adapter, unset_default_stt_adapter, get_default_stt_adapter,
)
from voice import rvc_client, stt, stt_adapter_infer
from voice.scripts import get_scripts
from engine import voice_engine, realism_engine, stt_train_engine
from engine.server_log import read_recent_lines, get_logger

logger = get_logger()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.makedirs(VOICE_SAMPLES_DIR, exist_ok=True)
os.makedirs(STT_SAMPLES_DIR, exist_ok=True)

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


# CSRF: the dashboard's session cookie defaults to Starlette's SameSite=Lax, which
# already blocks the browser from attaching it to most cross-site state-changing
# requests — this token is defense-in-depth on top of that, checked here (the one
# dependency every /manager/* route already uses) rather than per-route, and only
# for methods that actually change state. Login mints it into the session (see
# login_submit); dashboard.html echoes it back on every fetch() via the api() helper.
CSRF_HEADER = "X-CSRF-Token"
_CSRF_SAFE_METHODS = {"GET", "HEAD", "OPTIONS"}


def require_manager(request: Request):
    if not manager_logged_in(request):
        raise HTTPException(status_code=401, detail="Not logged in")
    if request.method not in _CSRF_SAFE_METHODS:
        token = request.headers.get(CSRF_HEADER)
        if not token or token != request.session.get("csrf_token"):
            raise HTTPException(status_code=403, detail="Thiếu hoặc sai CSRF token — vui lòng tải lại trang.")
    return request.session["manager"]


# Login rate limiting: in-process only (a single-worker deployment, matching this
# service's thesis scale) — tracks recent failed attempts per (scope, IP) and locks
# out further tries for a cooldown window. `scope` keeps the manager dashboard's
# counter separate from the STT Lab guest counter(s) sharing this same dict. A
# multi-worker/multi-instance deployment would need a shared store (e.g. Redis)
# instead of this module-level dict.
LOGIN_MAX_ATTEMPTS = 5
LOGIN_LOCKOUT_SEC  = 300  # 5 minutes
_failed_logins: dict[str, list[float]] = {}


def _client_ip(request: Request) -> str:
    return request.client.host if request.client else "unknown"


def _login_locked_out(scope: str, ip: str, max_attempts: int = LOGIN_MAX_ATTEMPTS,
                       window_sec: int = LOGIN_LOCKOUT_SEC) -> bool:
    key = f"{scope}:{ip}"
    now = time.time()
    recent = [t for t in _failed_logins.get(key, []) if now - t < window_sec]
    _failed_logins[key] = recent
    return len(recent) >= max_attempts


def _record_failed_login(scope: str, ip: str):
    _failed_logins.setdefault(f"{scope}:{ip}", []).append(time.time())


def _owns_cloned(profile: dict, client_id: int, external_user_id: str) -> bool:
    return bool(
        profile and profile["kind"] == "cloned"
        and profile["client_id"] == client_id
        and profile["external_user_id"] == external_user_id
    )


# ── Auth: STT Lab guest accounts (session, separate from both manager and
# client-app auth — a guest arrives with no host app and no API key) ───────────
REGISTER_MAX_ATTEMPTS = 10
REGISTER_LOCKOUT_SEC  = 3600  # 1 hour, per IP — blunts mass account creation
STT_USERNAME_RE = re.compile(r"^[a-zA-Z0-9_.-]{3,32}$")
STT_MIN_PASSWORD_LEN = 8


def stt_guest_logged_in(request: Request) -> bool:
    return bool(request.session.get("stt_guest_id"))


def require_stt_guest(request: Request) -> int:
    if not stt_guest_logged_in(request):
        raise HTTPException(status_code=401, detail="Not logged in")
    if request.method not in _CSRF_SAFE_METHODS:
        token = request.headers.get(CSRF_HEADER)
        if not token or token != request.session.get("stt_csrf_token"):
            raise HTTPException(status_code=403, detail="Thiếu hoặc sai CSRF token — vui lòng tải lại trang.")
    return request.session["stt_guest_id"]


def _owns_stt_adapter(adapter: dict, guest_id: int) -> bool:
    return bool(adapter and adapter["guest_id"] == guest_id)


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
    ip = _client_ip(request)
    if _login_locked_out("manager", ip):
        return templates.TemplateResponse(
            request, "login.html",
            {"error": "Quá nhiều lần đăng nhập sai — vui lòng thử lại sau 5 phút."}, status_code=429,
        )

    if not verify_manager_login(username, password):
        _record_failed_login("manager", ip)
        return templates.TemplateResponse(
            request, "login.html",
            {"error": "Sai tên đăng nhập hoặc mật khẩu."}, status_code=401,
        )

    _failed_logins.pop(f"manager:{ip}", None)
    # Clear any leftover STT Lab guest identity from this same browser session (added
    # 2026-08-12) -- manager and guest identity live in the same session cookie, so
    # without this, a manager who'd separately logged into /stt-lab as some guest
    # earlier in this browser would still resolve as that guest on /stt-lab after
    # also logging in here, i.e. the manager dashboard "leaking into" a specific
    # guest's own page. Manager oversight of guest adapters already has its own path
    # (/manager/stt, require_manager) that doesn't need this coexistence.
    request.session.pop("stt_guest_id", None)
    request.session.pop("stt_guest_username", None)
    request.session.pop("stt_csrf_token", None)
    request.session["manager"] = username
    request.session["csrf_token"] = secrets.token_urlsafe(32)
    return RedirectResponse("/", status_code=302)


@app.post("/logout")
async def logout_route(request: Request):
    request.session.clear()
    return RedirectResponse("/login", status_code=302)


@app.get("/", response_class=HTMLResponse)
async def dashboard_page(request: Request):
    if not manager_logged_in(request):
        return RedirectResponse("/login", status_code=302)
    # Sessions created before CSRF protection was added won't have a token yet --
    # mint one now rather than forcing an explicit re-login.
    if not request.session.get("csrf_token"):
        request.session["csrf_token"] = secrets.token_urlsafe(32)
    return templates.TemplateResponse(request, "dashboard.html", {
        "manager": request.session["manager"],
        "csrf_token": request.session["csrf_token"],
        "min_samples": MIN_TRAIN_SAMPLES,
    })


@app.get("/manager/stt", response_class=HTMLResponse)
async def manager_stt_page(request: Request):
    if not manager_logged_in(request):
        return RedirectResponse("/login", status_code=302)
    if not request.session.get("csrf_token"):
        request.session["csrf_token"] = secrets.token_urlsafe(32)
    return templates.TemplateResponse(request, "manager_stt.html", {
        "manager": request.session["manager"],
        "csrf_token": request.session["csrf_token"],
        "max_hotwords": MAX_HOTWORDS_PER_ADAPTER,
        "max_stt_train_samples": MAX_STT_TRAIN_SAMPLES,
        "min_stt_train_samples": MIN_STT_TRAIN_SAMPLES,
    })


# ── STT Lab: guest self-serve hotword adapters — register/login/logout/page ────
@app.get("/stt-lab/register", response_class=HTMLResponse)
async def stt_register_page(request: Request):
    if stt_guest_logged_in(request):
        return RedirectResponse("/stt-lab", status_code=302)
    return templates.TemplateResponse(request, "stt_guest_register.html", {"error": None})


@app.post("/stt-lab/register", response_class=HTMLResponse)
async def stt_register_submit(request: Request, username: str = Form(...), password: str = Form(...)):
    ip = _client_ip(request)
    if _login_locked_out("stt_register", ip, REGISTER_MAX_ATTEMPTS, REGISTER_LOCKOUT_SEC):
        return templates.TemplateResponse(
            request, "stt_guest_register.html",
            {"error": "Quá nhiều lần đăng ký — vui lòng thử lại sau 1 giờ."}, status_code=429,
        )

    if not STT_USERNAME_RE.match(username):
        return templates.TemplateResponse(
            request, "stt_guest_register.html",
            {"error": "Tên đăng nhập phải dài 3-32 ký tự, chỉ gồm chữ/số/_/./-"}, status_code=400,
        )
    if len(password) < STT_MIN_PASSWORD_LEN:
        return templates.TemplateResponse(
            request, "stt_guest_register.html",
            {"error": f"Mật khẩu phải có ít nhất {STT_MIN_PASSWORD_LEN} ký tự."}, status_code=400,
        )

    try:
        guest_id = create_stt_guest(username, password)
    except sqlite3.IntegrityError:
        _record_failed_login("stt_register", ip)
        return templates.TemplateResponse(
            request, "stt_guest_register.html",
            {"error": "Tên đăng nhập đã được sử dụng."}, status_code=409,
        )

    # Symmetric with login_submit's clear above -- a manager identity left over in
    # this same browser session must not coexist with a freshly registered guest.
    request.session.pop("manager", None)
    request.session.pop("csrf_token", None)
    request.session["stt_guest_id"] = guest_id
    request.session["stt_guest_username"] = username
    request.session["stt_csrf_token"] = secrets.token_urlsafe(32)
    return RedirectResponse("/stt-lab", status_code=302)


@app.get("/stt-lab/login", response_class=HTMLResponse)
async def stt_login_page(request: Request):
    if stt_guest_logged_in(request):
        return RedirectResponse("/stt-lab", status_code=302)
    return templates.TemplateResponse(request, "stt_guest_login.html", {"error": None})


@app.post("/stt-lab/login", response_class=HTMLResponse)
async def stt_login_submit(request: Request, username: str = Form(...), password: str = Form(...)):
    ip = _client_ip(request)
    if _login_locked_out("stt", ip):
        return templates.TemplateResponse(
            request, "stt_guest_login.html",
            {"error": "Quá nhiều lần đăng nhập sai — vui lòng thử lại sau 5 phút."}, status_code=429,
        )

    guest = verify_stt_guest_login(username, password)
    if not guest:
        _record_failed_login("stt", ip)
        return templates.TemplateResponse(
            request, "stt_guest_login.html",
            {"error": "Sai tên đăng nhập hoặc mật khẩu."}, status_code=401,
        )

    _failed_logins.pop(f"stt:{ip}", None)
    # Same clear as registration above -- see login_submit's comment for why.
    request.session.pop("manager", None)
    request.session.pop("csrf_token", None)
    request.session["stt_guest_id"] = guest["id"]
    request.session["stt_guest_username"] = guest["username"]
    request.session["stt_csrf_token"] = secrets.token_urlsafe(32)
    return RedirectResponse("/stt-lab", status_code=302)


@app.post("/stt-lab/logout")
async def stt_logout_route(request: Request):
    # Pop only this feature's session keys — a manager and a guest logged in
    # from the same browser must not log each other out (see require_manager's
    # own `request.session["manager"]` sharing this same session cookie).
    request.session.pop("stt_guest_id", None)
    request.session.pop("stt_guest_username", None)
    request.session.pop("stt_csrf_token", None)
    return RedirectResponse("/stt-lab/login", status_code=302)


@app.get("/stt-lab", response_class=HTMLResponse)
async def stt_dashboard_page(request: Request):
    if not stt_guest_logged_in(request):
        return RedirectResponse("/stt-lab/login", status_code=302)
    if not request.session.get("stt_csrf_token"):
        request.session["stt_csrf_token"] = secrets.token_urlsafe(32)
    return templates.TemplateResponse(request, "stt_guest_dashboard.html", {
        "guest_username": request.session["stt_guest_username"],
        "csrf_token": request.session["stt_csrf_token"],
        "max_hotwords": MAX_HOTWORDS_PER_ADAPTER,
        "max_hotword_len": MAX_HOTWORD_LEN,
        "max_stt_adapters": MAX_STT_ADAPTERS_PER_GUEST,
        "max_stt_train_samples": MAX_STT_TRAIN_SAMPLES,
        "min_stt_train_samples": MIN_STT_TRAIN_SAMPLES,
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


# ── Manager dashboard: server log ───────────────────────────────────────────────
@app.get("/manager/logs")
async def get_server_logs_route(manager: str = Depends(require_manager)):
    """Recent application-level events (training progress, Colab/local fallback
    decisions, errors — see engine/server_log.py) so a manager can see what's
    happening without watching the raw terminal. Not the full process stdout/
    stderr or uvicorn's own per-request access log — those are noise for this
    purpose; just this app's own logger.info()/warning()/error() calls."""
    return {"lines": read_recent_lines(500)}


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

    If a Tier 2 STT adapter has been published for this client (self-serve,
    see /api/stt/adapters/{id}/publish), tries that fine-tuned model first.
    Otherwise, tries the manager's system-wide default adapter if one is set
    (see /manager/stt/adapters/{id}/set-default) -- a manager-owned adapter
    that isn't tied to any single client_id. Otherwise -- or if either adapter
    fails to load/run -- tries the Colab-hosted PhoWhisper endpoint
    (Vietnamese-tuned, see voice/rvc_client.py's transcribe_remote() and
    colab/voice_server.ipynb) and falls back to the local, CPU-only
    openai-whisper model (voice/stt.py) whenever Colab is unset/unreachable —
    same degrade-gracefully contract as /api/speak's RVC conversion."""
    audio_bytes = await audio.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Không có dữ liệu âm thanh.")

    mime = audio.content_type or mimetypes.guess_type(audio.filename or "")[0] or "audio/webm"

    result = None
    adapter = get_published_stt_adapter_for_client(client["id"]) or get_default_stt_adapter()
    if adapter:
        try:
            result = await asyncio.to_thread(
                stt_adapter_infer.transcribe, adapter, audio_bytes, mime, language or None
            )
        except Exception as e:
            logger.warning(f"[STT-adapter] Adapter #{adapter['id']} lỗi, dùng model gốc thay thế: {e}")

    if result is None:
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


# ── STT Lab: guest-owned hotword/prompt-bias adapters (Tier 1) ─────────────────
def _clean_hotwords(raw) -> list:
    if not isinstance(raw, list):
        raise HTTPException(status_code=400, detail="hotwords phải là một danh sách chuỗi.")
    if len(raw) > MAX_HOTWORDS_PER_ADAPTER:
        raise HTTPException(status_code=400, detail=f"Tối đa {MAX_HOTWORDS_PER_ADAPTER} từ khoá.")
    cleaned = []
    seen = set()
    for w in raw:
        if not isinstance(w, str):
            raise HTTPException(status_code=400, detail="Mỗi từ khoá phải là chuỗi.")
        w = w.strip()
        if not w or len(w) > MAX_HOTWORD_LEN:
            raise HTTPException(status_code=400, detail=f"Mỗi từ khoá dài 1-{MAX_HOTWORD_LEN} ký tự.")
        if w not in seen:
            seen.add(w)
            cleaned.append(w)
    return cleaned


def _get_owned_stt_adapter(adapter_id: int, guest_id: int) -> dict:
    adapter = get_stt_adapter(adapter_id)
    if not _owns_stt_adapter(adapter, guest_id):
        raise HTTPException(status_code=404, detail="Không tìm thấy adapter.")
    return adapter


def _build_stt_pack_response(adapter: dict) -> Response:
    """Builds the .stt-pack.zip Response for an adapter -- shared by the
    guest's own download, the manager's any-adapter download, and the
    guest-facing "download the current system default" route below (three
    different auth/ownership rules, same file format)."""
    has_lora = bool(adapter["adapter_path"]) and os.path.isdir(adapter["adapter_path"])
    manifest = {
        "adapter_id": adapter["id"],
        "name": adapter["name"],
        "base_model": adapter["base_model"],
        "tier": 2 if has_lora else 1,
        "backend_used": adapter["backend_used"] if has_lora else None,
        "created_at": adapter["created_at"],
    }
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))
        zf.writestr("hotwords.json", json.dumps(adapter["hotwords"], ensure_ascii=False, indent=2))
        if has_lora:
            for fname in ("adapter_model.safetensors", "adapter_config.json"):
                fpath = os.path.join(adapter["adapter_path"], fname)
                if os.path.exists(fpath):
                    zf.write(fpath, arcname=fname)

    return Response(
        content=buf.getvalue(),
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{adapter["id"]}.stt-pack.zip"'},
    )


def _reject_if_training(adapter: dict):
    """Guards every hotwords/sample mutation below -- the frontend already disables
    these controls while status=='training' (see stt_guest_dashboard.html), but that
    alone doesn't stop a direct API call, and editing the data out from under an
    in-progress local/Colab training run is exactly the confusing case a guest asked
    this to prevent."""
    if adapter.get("status") == "training":
        raise HTTPException(status_code=400, detail="Adapter đang huấn luyện -- không thể sửa lúc này.")


@app.post("/api/stt/adapters")
async def create_stt_adapter_route(request: Request, guest_id: int = Depends(require_stt_guest)):
    data = await request.json()
    name = (data.get("name") or "").strip()
    if not name or len(name) > 100:
        raise HTTPException(status_code=400, detail="Tên adapter phải dài 1-100 ký tự.")
    base_model = data.get("base_model") or ALLOWED_STT_BASE_MODELS[0]
    if base_model not in ALLOWED_STT_BASE_MODELS:
        raise HTTPException(status_code=400, detail=f"base_model phải là một trong: {', '.join(ALLOWED_STT_BASE_MODELS)}")
    if count_stt_adapters(guest_id) >= MAX_STT_ADAPTERS_PER_GUEST:
        raise HTTPException(status_code=400, detail=f"Tối đa {MAX_STT_ADAPTERS_PER_GUEST} adapter mỗi tài khoản.")
    adapter_id = create_stt_adapter(guest_id, name, base_model)
    return {"status": "ok", "adapter_id": adapter_id}


@app.get("/api/stt/adapters")
async def list_stt_adapters_route(guest_id: int = Depends(require_stt_guest)):
    return list_stt_adapters(guest_id)


@app.get("/api/stt/adapters/{adapter_id}")
async def get_stt_adapter_route(adapter_id: int, guest_id: int = Depends(require_stt_guest)):
    return _get_owned_stt_adapter(adapter_id, guest_id)


@app.put("/api/stt/adapters/{adapter_id}")
async def update_stt_adapter_route(adapter_id: int, request: Request, guest_id: int = Depends(require_stt_guest)):
    adapter = _get_owned_stt_adapter(adapter_id, guest_id)
    data = await request.json()
    if "name" in data:
        name = (data.get("name") or "").strip()
        if not name or len(name) > 100:
            raise HTTPException(status_code=400, detail="Tên adapter phải dài 1-100 ký tự.")
        rename_stt_adapter(adapter_id, name)
    if "hotwords" in data:
        _reject_if_training(adapter)
        update_stt_adapter_hotwords(adapter_id, _clean_hotwords(data.get("hotwords")))
    return {"status": "ok"}


# ── STT Lab: training samples + Tier 2 (LoRA fine-tune) ────────────────────────
@app.post("/api/stt/adapters/{adapter_id}/samples")
async def upload_stt_sample_route(adapter_id: int, reference_text: str = Form(...),
                                   audio: UploadFile = File(...), is_holdout: bool = Form(False),
                                   guest_id: int = Depends(require_stt_guest)):
    adapter = _get_owned_stt_adapter(adapter_id, guest_id)
    _reject_if_training(adapter)
    reference_text = reference_text.strip()
    if not reference_text:
        raise HTTPException(status_code=400, detail="Cần nhập transcript cho mẫu ghi âm.")
    if len(list_stt_training_samples(adapter_id)) >= MAX_STT_TRAIN_SAMPLES:
        raise HTTPException(status_code=400, detail=f"Tối đa {MAX_STT_TRAIN_SAMPLES} mẫu mỗi adapter.")

    content = await audio.read()
    ext = os.path.splitext(audio.filename or "")[1] or ".wav"
    sample_dir = os.path.join(STT_SAMPLES_DIR, str(adapter_id))
    os.makedirs(sample_dir, exist_ok=True)
    sample_path = os.path.join(sample_dir, f"{uuid.uuid4()}{ext}")
    with open(sample_path, "wb") as f:
        f.write(content)

    try:
        info = sf.info(sample_path)
        duration = info.frames / info.samplerate
    except Exception:
        os.remove(sample_path)
        raise HTTPException(status_code=400, detail="Không đọc được file âm thanh.")
    if duration > MAX_STT_SAMPLE_DURATION_SEC:
        os.remove(sample_path)
        raise HTTPException(status_code=400, detail=f"Mẫu ghi âm tối đa {MAX_STT_SAMPLE_DURATION_SEC} giây.")

    sample_id = add_stt_training_sample(adapter_id, sample_path, reference_text, is_holdout=is_holdout)
    return {"status": "ok", "sample_id": sample_id}


@app.get("/api/stt/adapters/{adapter_id}/samples")
async def list_stt_samples_route(adapter_id: int, guest_id: int = Depends(require_stt_guest)):
    _get_owned_stt_adapter(adapter_id, guest_id)
    return list_stt_training_samples(adapter_id)


@app.get("/api/stt/adapters/{adapter_id}/samples/{sample_id}/audio")
async def stt_sample_audio_route(adapter_id: int, sample_id: int, guest_id: int = Depends(require_stt_guest)):
    _get_owned_stt_adapter(adapter_id, guest_id)
    sample = get_stt_training_sample(sample_id)
    if not sample or sample["adapter_id"] != adapter_id or not os.path.exists(sample["audio_path"]):
        raise HTTPException(status_code=404, detail="Không tìm thấy file mẫu ghi âm.")
    mime = mimetypes.guess_type(sample["audio_path"])[0] or "application/octet-stream"
    return FileResponse(sample["audio_path"], media_type=mime)


@app.put("/api/stt/adapters/{adapter_id}/samples/{sample_id}")
async def update_stt_sample_route(adapter_id: int, sample_id: int, request: Request,
                                   guest_id: int = Depends(require_stt_guest)):
    adapter = _get_owned_stt_adapter(adapter_id, guest_id)
    _reject_if_training(adapter)
    sample = get_stt_training_sample(sample_id)
    if not sample or sample["adapter_id"] != adapter_id:
        raise HTTPException(status_code=404, detail="Không tìm thấy mẫu.")
    data = await request.json()
    reference_text = (data.get("reference_text") or "").strip()
    if not reference_text:
        raise HTTPException(status_code=400, detail="Cần nhập transcript cho mẫu ghi âm.")
    update_stt_training_sample_text(sample_id, reference_text)
    return {"status": "ok"}


@app.delete("/api/stt/adapters/{adapter_id}/samples/{sample_id}")
async def delete_stt_sample_route(adapter_id: int, sample_id: int, guest_id: int = Depends(require_stt_guest)):
    adapter = _get_owned_stt_adapter(adapter_id, guest_id)
    _reject_if_training(adapter)
    sample = get_stt_training_sample(sample_id)
    if not sample or sample["adapter_id"] != adapter_id:
        raise HTTPException(status_code=404, detail="Không tìm thấy mẫu.")
    if os.path.exists(sample["audio_path"]):
        os.remove(sample["audio_path"])
    delete_stt_training_sample(sample_id)
    return {"status": "ok"}


@app.post("/api/stt/adapters/{adapter_id}/train")
async def train_stt_adapter_route(adapter_id: int, request: Request, background_tasks: BackgroundTasks,
                                   guest_id: int = Depends(require_stt_guest)):
    adapter = _get_owned_stt_adapter(adapter_id, guest_id)
    if adapter["status"] == "training":
        raise HTTPException(status_code=400, detail="Adapter này đang được huấn luyện.")
    if len(list_stt_training_samples(adapter_id)) < MIN_STT_TRAIN_SAMPLES:
        raise HTTPException(status_code=400, detail=f"Cần tối thiểu {MIN_STT_TRAIN_SAMPLES} mẫu để huấn luyện.")

    data = await request.json() if await request.body() else {}
    backend = data.get("backend", "auto")
    if backend not in ("auto", "colab", "local"):
        raise HTTPException(status_code=400, detail="backend phải là auto | colab | local.")

    background_tasks.add_task(stt_train_engine.run_training, adapter_id, backend)
    return {"status": "ok", "message": "Đã bắt đầu huấn luyện."}


@app.post("/api/stt/adapters/{adapter_id}/continue")
async def continue_stt_adapter_route(adapter_id: int, pack: UploadFile = File(...),
                                      guest_id: int = Depends(require_stt_guest)):
    adapter = _get_owned_stt_adapter(adapter_id, guest_id)
    content = await pack.read()

    try:
        with zipfile.ZipFile(io.BytesIO(content)) as zf:
            manifest = json.loads(zf.read("manifest.json"))
            if manifest.get("tier") != 2:
                raise HTTPException(status_code=400, detail="Pack này không chứa adapter đã huấn luyện (Tier 2).")
            if manifest.get("base_model") != adapter["base_model"]:
                raise HTTPException(
                    status_code=400,
                    detail=f"Pack này dùng base_model={manifest.get('base_model')}, adapter hiện tại dùng {adapter['base_model']}.",
                )
            resume_dir = os.path.join(STT_MODELS_DIR, f"{adapter_id}_resume")
            if os.path.isdir(resume_dir):
                shutil.rmtree(resume_dir)
            os.makedirs(resume_dir, exist_ok=True)
            zf.extract("adapter_model.safetensors", resume_dir)
            zf.extract("adapter_config.json", resume_dir)
    except KeyError:
        raise HTTPException(status_code=400, detail="Pack thiếu adapter_model.safetensors hoặc adapter_config.json.")
    except zipfile.BadZipFile:
        raise HTTPException(status_code=400, detail="File không phải zip hợp lệ.")

    set_stt_adapter_resume_path(adapter_id, resume_dir)
    return {"status": "ok", "message": "Đã nạp adapter cũ — lần huấn luyện tiếp theo sẽ tiếp tục từ đây."}


@app.get("/api/stt/adapters/{adapter_id}/download")
async def download_stt_adapter_route(adapter_id: int, guest_id: int = Depends(require_stt_guest)):
    adapter = _get_owned_stt_adapter(adapter_id, guest_id)
    return _build_stt_pack_response(adapter)


@app.get("/api/stt/default_pack/download")
async def download_default_stt_pack_route(guest_id: int = Depends(require_stt_guest)):
    """Lets any logged-in STT Lab guest grab the current system-wide default
    adapter (see /manager/stt/adapters/{id}/set-default) as a .stt-pack.zip --
    unlike the owner-scoped download above, this is deliberately NOT
    ownership-restricted, since the whole point of a system default is that
    it's already what /api/transcribe falls back to for every client with
    nothing of their own published; a guest wanting to try it locally
    (clone_voice_client.local_stt, see voice-lab-example's /settings) has no
    other way to find/download it if they don't happen to own it themselves."""
    adapter = get_default_stt_adapter()
    if not adapter:
        raise HTTPException(status_code=404, detail="Hệ thống chưa có adapter mặc định nào.")
    return _build_stt_pack_response(adapter)


@app.post("/api/stt/adapters/{adapter_id}/publish")
async def publish_stt_adapter_route(adapter_id: int, guest_id: int = Depends(require_stt_guest)):
    """Self-serve publish (added 2026-08-12): a guest activates their own ready adapter
    for their own auto-provisioned client's /api/transcribe traffic immediately, no
    manager step in between -- see _provision_stt_guest_client (database.py). This is
    the ONLY way to publish an adapter to a specific client -- there is deliberately no
    manager-side equivalent that picks an arbitrary client_id, so client_id is never
    taken from the request here either -- it's always the guest's own, read fresh from
    the DB (not the session) so a manager fix-up (see database.py) is honored right
    away. A manager can only revoke (POST /manager/stt/adapters/{id}/unpublish) or set
    a client-independent system default (.../set-default)."""
    adapter = _get_owned_stt_adapter(adapter_id, guest_id)
    if adapter["status"] != "ready" or not adapter["adapter_path"]:
        raise HTTPException(status_code=400,
            detail="Chỉ có thể publish adapter đã huấn luyện xong (status='ready').")
    guest = get_stt_guest(guest_id)
    if not guest or not guest["client_id"]:
        raise HTTPException(status_code=400,
            detail="Tài khoản của bạn chưa có client — vui lòng thử lại hoặc liên hệ quản trị viên.")
    publish_stt_adapter(adapter_id, guest["client_id"])
    return {"status": "ok"}


@app.post("/api/stt/adapters/{adapter_id}/unpublish")
async def unpublish_stt_adapter_route(adapter_id: int, guest_id: int = Depends(require_stt_guest)):
    _get_owned_stt_adapter(adapter_id, guest_id)
    unpublish_stt_adapter(adapter_id)
    return {"status": "ok"}


def _delete_stt_adapter_files(adapter: dict):
    """Removes this adapter's training-sample audio files and trained-model
    directory from disk — the DB-layer delete_stt_adapter()/
    delete_stt_guest_account() only drop rows, by design (no filesystem
    access in database.py), so this must run first while the paths are
    still known."""
    for sample in list_stt_training_samples(adapter["id"]):
        if os.path.exists(sample["audio_path"]):
            os.remove(sample["audio_path"])
    sample_dir = os.path.join(STT_SAMPLES_DIR, str(adapter["id"]))
    if os.path.isdir(sample_dir):
        shutil.rmtree(sample_dir, ignore_errors=True)
    if adapter["adapter_path"] and os.path.isdir(adapter["adapter_path"]):
        shutil.rmtree(adapter["adapter_path"], ignore_errors=True)
    resume_dir = os.path.join(STT_MODELS_DIR, f"{adapter['id']}_resume")
    if os.path.isdir(resume_dir):
        shutil.rmtree(resume_dir, ignore_errors=True)


@app.delete("/api/stt/adapters/{adapter_id}")
async def delete_stt_adapter_route(adapter_id: int, guest_id: int = Depends(require_stt_guest)):
    adapter = _get_owned_stt_adapter(adapter_id, guest_id)
    _delete_stt_adapter_files(adapter)
    delete_stt_adapter(adapter_id)
    return {"status": "ok"}


@app.delete("/api/stt/account")
async def delete_stt_account_route(request: Request, guest_id: int = Depends(require_stt_guest)):
    for adapter in list_stt_adapters(guest_id):
        _delete_stt_adapter_files(adapter)
    delete_stt_guest_account(guest_id)
    request.session.pop("stt_guest_id", None)
    request.session.pop("stt_guest_username", None)
    request.session.pop("stt_csrf_token", None)
    return {"status": "ok"}


# ── Manager: STT Tier 2 training + publish-to-client ────────────────────────
# Mirrors the guest STT Lab routes above (same underlying database.py functions,
# same engine/stt_train_engine.py training path, same _reject_if_training/
# _clean_hotwords/_delete_stt_adapter_files helpers) but scoped to require_manager
# instead of a guest's own ownership -- a manager can touch ANY adapter (own or
# any guest's), and additionally publish one as a client's active production
# model for /api/transcribe (see that route below). Kept as separate routes
# under /manager/stt/* rather than widening the guest routes' auth, so a guest's
# own self-serve capability and a manager's global one stay independently
# reasoned about, same split /manager/profiles/* already uses for RVC voices.
def _get_stt_adapter_or_404(adapter_id: int) -> dict:
    adapter = get_stt_adapter(adapter_id)
    if not adapter:
        raise HTTPException(status_code=404, detail="Không tìm thấy adapter.")
    return adapter


@app.get("/manager/stt/adapters")
async def manager_list_stt_adapters_route(manager: str = Depends(require_manager)):
    return list_all_stt_adapters_global()


@app.post("/manager/stt/adapters")
async def manager_create_stt_adapter_route(request: Request, manager: str = Depends(require_manager)):
    data = await request.json()
    name = (data.get("name") or "").strip()
    if not name or len(name) > 100:
        raise HTTPException(status_code=400, detail="Tên adapter phải dài 1-100 ký tự.")
    base_model = data.get("base_model") or ALLOWED_STT_BASE_MODELS[0]
    if base_model not in ALLOWED_STT_BASE_MODELS:
        raise HTTPException(status_code=400, detail=f"base_model phải là một trong: {', '.join(ALLOWED_STT_BASE_MODELS)}")
    # No MAX_STT_ADAPTERS_PER_GUEST cap here -- that limit exists to bound an
    # anonymous self-serve guest's GPU usage; a manager-created adapter is a
    # deliberate admin action, not drive-by usage.
    adapter_id = create_stt_adapter(None, name, base_model)
    return {"status": "ok", "adapter_id": adapter_id}


@app.get("/manager/stt/adapters/{adapter_id}")
async def manager_get_stt_adapter_route(adapter_id: int, manager: str = Depends(require_manager)):
    return _get_stt_adapter_or_404(adapter_id)


@app.put("/manager/stt/adapters/{adapter_id}")
async def manager_update_stt_adapter_route(adapter_id: int, request: Request, manager: str = Depends(require_manager)):
    adapter = _get_stt_adapter_or_404(adapter_id)
    data = await request.json()
    if "name" in data:
        name = (data.get("name") or "").strip()
        if not name or len(name) > 100:
            raise HTTPException(status_code=400, detail="Tên adapter phải dài 1-100 ký tự.")
        rename_stt_adapter(adapter_id, name)
    if "hotwords" in data:
        _reject_if_training(adapter)
        update_stt_adapter_hotwords(adapter_id, _clean_hotwords(data.get("hotwords")))
    return {"status": "ok"}


@app.post("/manager/stt/adapters/{adapter_id}/samples")
async def manager_upload_stt_sample_route(adapter_id: int, reference_text: str = Form(...),
                                           audio: UploadFile = File(...), is_holdout: bool = Form(False),
                                           manager: str = Depends(require_manager)):
    adapter = _get_stt_adapter_or_404(adapter_id)
    _reject_if_training(adapter)
    reference_text = reference_text.strip()
    if not reference_text:
        raise HTTPException(status_code=400, detail="Cần nhập transcript cho mẫu ghi âm.")
    if len(list_stt_training_samples(adapter_id)) >= MAX_STT_TRAIN_SAMPLES:
        raise HTTPException(status_code=400, detail=f"Tối đa {MAX_STT_TRAIN_SAMPLES} mẫu mỗi adapter.")

    content = await audio.read()
    ext = os.path.splitext(audio.filename or "")[1] or ".wav"
    sample_dir = os.path.join(STT_SAMPLES_DIR, str(adapter_id))
    os.makedirs(sample_dir, exist_ok=True)
    sample_path = os.path.join(sample_dir, f"{uuid.uuid4()}{ext}")
    with open(sample_path, "wb") as f:
        f.write(content)

    try:
        info = sf.info(sample_path)
        duration = info.frames / info.samplerate
    except Exception:
        os.remove(sample_path)
        raise HTTPException(status_code=400, detail="Không đọc được file âm thanh.")
    if duration > MAX_STT_SAMPLE_DURATION_SEC:
        os.remove(sample_path)
        raise HTTPException(status_code=400, detail=f"Mẫu ghi âm tối đa {MAX_STT_SAMPLE_DURATION_SEC} giây.")

    sample_id = add_stt_training_sample(adapter_id, sample_path, reference_text, is_holdout=is_holdout)
    return {"status": "ok", "sample_id": sample_id}


@app.get("/manager/stt/adapters/{adapter_id}/samples")
async def manager_list_stt_samples_route(adapter_id: int, manager: str = Depends(require_manager)):
    _get_stt_adapter_or_404(adapter_id)
    return list_stt_training_samples(adapter_id)


@app.get("/manager/stt/adapters/{adapter_id}/samples/{sample_id}/audio")
async def manager_stt_sample_audio_route(adapter_id: int, sample_id: int, manager: str = Depends(require_manager)):
    _get_stt_adapter_or_404(adapter_id)
    sample = get_stt_training_sample(sample_id)
    if not sample or sample["adapter_id"] != adapter_id or not os.path.exists(sample["audio_path"]):
        raise HTTPException(status_code=404, detail="Không tìm thấy file mẫu ghi âm.")
    mime = mimetypes.guess_type(sample["audio_path"])[0] or "application/octet-stream"
    return FileResponse(sample["audio_path"], media_type=mime)


@app.put("/manager/stt/adapters/{adapter_id}/samples/{sample_id}")
async def manager_update_stt_sample_route(adapter_id: int, sample_id: int, request: Request,
                                           manager: str = Depends(require_manager)):
    adapter = _get_stt_adapter_or_404(adapter_id)
    _reject_if_training(adapter)
    sample = get_stt_training_sample(sample_id)
    if not sample or sample["adapter_id"] != adapter_id:
        raise HTTPException(status_code=404, detail="Không tìm thấy mẫu.")
    data = await request.json()
    reference_text = (data.get("reference_text") or "").strip()
    if not reference_text:
        raise HTTPException(status_code=400, detail="Cần nhập transcript cho mẫu ghi âm.")
    update_stt_training_sample_text(sample_id, reference_text)
    return {"status": "ok"}


@app.delete("/manager/stt/adapters/{adapter_id}/samples/{sample_id}")
async def manager_delete_stt_sample_route(adapter_id: int, sample_id: int, manager: str = Depends(require_manager)):
    adapter = _get_stt_adapter_or_404(adapter_id)
    _reject_if_training(adapter)
    sample = get_stt_training_sample(sample_id)
    if not sample or sample["adapter_id"] != adapter_id:
        raise HTTPException(status_code=404, detail="Không tìm thấy mẫu.")
    if os.path.exists(sample["audio_path"]):
        os.remove(sample["audio_path"])
    delete_stt_training_sample(sample_id)
    return {"status": "ok"}


@app.post("/manager/stt/adapters/{adapter_id}/train")
async def manager_train_stt_adapter_route(adapter_id: int, request: Request, background_tasks: BackgroundTasks,
                                           manager: str = Depends(require_manager)):
    adapter = _get_stt_adapter_or_404(adapter_id)
    if adapter["status"] == "training":
        raise HTTPException(status_code=400, detail="Adapter này đang được huấn luyện.")
    if len(list_stt_training_samples(adapter_id)) < MIN_STT_TRAIN_SAMPLES:
        raise HTTPException(status_code=400, detail=f"Cần tối thiểu {MIN_STT_TRAIN_SAMPLES} mẫu để huấn luyện.")

    data = await request.json() if await request.body() else {}
    backend = data.get("backend", "auto")
    if backend not in ("auto", "colab", "local"):
        raise HTTPException(status_code=400, detail="backend phải là auto | colab | local.")

    background_tasks.add_task(stt_train_engine.run_training, adapter_id, backend)
    return {"status": "ok", "message": "Đã bắt đầu huấn luyện."}


@app.post("/manager/stt/adapters/{adapter_id}/continue")
async def manager_continue_stt_adapter_route(adapter_id: int, pack: UploadFile = File(...),
                                              manager: str = Depends(require_manager)):
    adapter = _get_stt_adapter_or_404(adapter_id)
    content = await pack.read()

    try:
        with zipfile.ZipFile(io.BytesIO(content)) as zf:
            manifest = json.loads(zf.read("manifest.json"))
            if manifest.get("tier") != 2:
                raise HTTPException(status_code=400, detail="Pack này không chứa adapter đã huấn luyện (Tier 2).")
            if manifest.get("base_model") != adapter["base_model"]:
                raise HTTPException(
                    status_code=400,
                    detail=f"Pack này dùng base_model={manifest.get('base_model')}, adapter hiện tại dùng {adapter['base_model']}.",
                )
            resume_dir = os.path.join(STT_MODELS_DIR, f"{adapter_id}_resume")
            if os.path.isdir(resume_dir):
                shutil.rmtree(resume_dir)
            os.makedirs(resume_dir, exist_ok=True)
            zf.extract("adapter_model.safetensors", resume_dir)
            zf.extract("adapter_config.json", resume_dir)
    except KeyError:
        raise HTTPException(status_code=400, detail="Pack thiếu adapter_model.safetensors hoặc adapter_config.json.")
    except zipfile.BadZipFile:
        raise HTTPException(status_code=400, detail="File không phải zip hợp lệ.")

    set_stt_adapter_resume_path(adapter_id, resume_dir)
    return {"status": "ok", "message": "Đã nạp adapter cũ — lần huấn luyện tiếp theo sẽ tiếp tục từ đây."}


@app.get("/manager/stt/adapters/{adapter_id}/download")
async def manager_download_stt_adapter_route(adapter_id: int, manager: str = Depends(require_manager)):
    adapter = _get_stt_adapter_or_404(adapter_id)
    return _build_stt_pack_response(adapter)


@app.post("/manager/stt/adapters/{adapter_id}/unpublish")
async def manager_unpublish_stt_adapter_route(adapter_id: int, manager: str = Depends(require_manager)):
    """Revoke only -- a manager can un-publish (e.g. for moderation) but can't
    publish to an arbitrary client themselves; publishing is guest self-serve
    only, always onto the guest's own auto-provisioned client (see
    POST /api/stt/adapters/{id}/publish)."""
    _get_stt_adapter_or_404(adapter_id)
    unpublish_stt_adapter(adapter_id)
    return {"status": "ok"}


@app.post("/manager/stt/adapters/{adapter_id}/set-default")
async def manager_set_default_stt_adapter_route(adapter_id: int, manager: str = Depends(require_manager)):
    """Sets adapter_id as the system-wide fallback for any client with nothing of
    its own published -- unlike /publish, this takes no client_id at all."""
    adapter = _get_stt_adapter_or_404(adapter_id)
    if adapter["status"] != "ready" or not adapter["adapter_path"]:
        raise HTTPException(
            status_code=400,
            detail="Chỉ có thể đặt mặc định cho adapter đã huấn luyện xong (status='ready', có adapter_path).",
        )
    set_default_stt_adapter(adapter_id)
    return {"status": "ok"}


@app.post("/manager/stt/adapters/{adapter_id}/unset-default")
async def manager_unset_default_stt_adapter_route(adapter_id: int, manager: str = Depends(require_manager)):
    _get_stt_adapter_or_404(adapter_id)
    unset_default_stt_adapter(adapter_id)
    return {"status": "ok"}


@app.delete("/manager/stt/adapters/{adapter_id}")
async def manager_delete_stt_adapter_route(adapter_id: int, manager: str = Depends(require_manager)):
    adapter = _get_stt_adapter_or_404(adapter_id)
    _delete_stt_adapter_files(adapter)
    delete_stt_adapter(adapter_id)
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8090, reload=False)
