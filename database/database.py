# database.py

import hashlib
import json
import os
import secrets
import shutil
import sqlite3
import time

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # database/ → root
DB_NAME  = os.path.join(BASE_DIR, "voice_station.db")

VOICE_SAMPLES_DIR = os.path.join(BASE_DIR, "voice_samples")
VOICE_MODELS_DIR  = os.path.join(BASE_DIR, "voice_storage")

# (display name, engine voice id — "f5tts:" prefix routes to the F5-TTS-Vietnamese-
# ViVoice baseline on Colab instead of edge-TTS, see voice/tts.py). Shared across every
# client — builtin rows carry client_id/external_user_id = NULL.
BUILTIN_VOICES = [
    ("HoaiMy (Nữ)",  "vi-VN-HoaiMyNeural"),
    ("NamMinh (Nam)", "vi-VN-NamMinhNeural"),
    ("Jenny (EN)",   "en-US-JennyNeural"),
    ("F5-TTS demo (VN)", "f5tts:default"),
]

MIN_TRAIN_SAMPLES = 5

# Testing/demo-phase cap — keeps load on the free Colab GPU bounded until
# there's enough traction to justify paying for more training capacity.
# A client's end user must delete an old cloned voice before creating another one past this.
MAX_CLONED_VOICES_PER_USER = 2

# STT Lab (guest self-serve hotword adapters) — caps enforced server-side on
# every write, regardless of what the page's own JS already limits client-side.
MAX_HOTWORDS_PER_ADAPTER = 200
MAX_HOTWORD_LEN = 80

# First client seeded on a fresh DB — matches the app this service was extracted from.
DEFAULT_CLIENT_NAME = "rag-legal-assistant"

# Default account for the manager dashboard, seeded on first init only.
DEFAULT_MANAGER_USERNAME = "manager"

PBKDF2_ITERATIONS = 200_000


def get_conn():
    return sqlite3.connect(DB_NAME)


# =========================
# PASSWORD HASHING (manager accounts only — stdlib PBKDF2, no extra dependency)
# =========================
def _hash_password(password: str, salt: bytes = None) -> str:
    salt = salt or os.urandom(16)
    dk   = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, PBKDF2_ITERATIONS)
    return f"{salt.hex()}${dk.hex()}"


def _verify_password(password: str, stored: str) -> bool:
    try:
        salt_hex, hash_hex = stored.split("$")
    except (ValueError, AttributeError):
        return False
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), bytes.fromhex(salt_hex), PBKDF2_ITERATIONS)
    return secrets.compare_digest(dk.hex(), hash_hex)


# =========================
# INIT DATABASE
# =========================
def init_db():
    conn = sqlite3.connect(DB_NAME)
    c    = conn.cursor()

    c.execute("""
        CREATE TABLE IF NOT EXISTS clients (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            name        TEXT UNIQUE,
            api_key     TEXT UNIQUE,
            webhook_url TEXT,
            created_at  REAL
        )
    """)
    # Migration for existing DBs missing webhook_url column
    try:
        c.execute("ALTER TABLE clients ADD COLUMN webhook_url TEXT")
    except Exception:
        pass

    # kind:   'builtin' = system default edge-TTS voice (client_id/external_user_id NULL)
    #         'cloned'  = one end user's personal voice, trained via RVC
    c.execute("""
        CREATE TABLE IF NOT EXISTS voice_profiles (
            id                INTEGER PRIMARY KEY AUTOINCREMENT,
            client_id         INTEGER,
            external_user_id  TEXT,
            name              TEXT,
            kind              TEXT DEFAULT 'cloned',
            base_tts_voice    TEXT,
            speaker_id        TEXT,
            status            TEXT DEFAULT 'new',
            is_default        INTEGER DEFAULT 0,
            error_message     TEXT,
            model_local_path  TEXT,
            created_at        REAL
        )
    """)
    # Migration for existing DBs missing progress_message (live status text shown
    # in the frontend while status='training' -- see engine/voice_engine.py's
    # progress_cb and GET /api/profiles/{id}/status).
    try:
        c.execute("ALTER TABLE voice_profiles ADD COLUMN progress_message TEXT")
    except Exception:
        pass

    c.execute("""
        CREATE TABLE IF NOT EXISTS voice_samples (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            profile_id  INTEGER,
            script_id   TEXT,
            file_path   TEXT,
            created_at  REAL
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS voice_consent (
            client_id         INTEGER,
            external_user_id  TEXT,
            consented_at      REAL,
            PRIMARY KEY (client_id, external_user_id)
        )
    """)

    # Fired when a MANAGER (not the end user themselves, not a client app's own
    # admin) deletes or disables a cloned voice — so the end user can be told
    # why their voice stopped working. Delivered primarily via each client's
    # registered webhook_url (see clients.webhook_url); `delivered_at` stays
    # NULL until that POST succeeds, so GET /api/notifications can serve any
    # still-undelivered ones as a polling fallback.
    c.execute("""
        CREATE TABLE IF NOT EXISTS notifications (
            id                INTEGER PRIMARY KEY AUTOINCREMENT,
            client_id         INTEGER,
            external_user_id  TEXT,
            profile_id        INTEGER,
            profile_name      TEXT,
            event             TEXT,
            message           TEXT,
            created_at        REAL,
            delivered_at      REAL
        )
    """)

    # Generic key/value store, e.g. "rvc_endpoint" (the Colab tunnel URL).
    c.execute("""
        CREATE TABLE IF NOT EXISTS settings (
            key   TEXT PRIMARY KEY,
            value TEXT
        )
    """)

    # Manager dashboard accounts — entirely separate from any client app's own
    # users. Whoever holds these credentials can see/manage every client and
    # every end user's voice profiles, so keep this list short.
    c.execute("""
        CREATE TABLE IF NOT EXISTS managers (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            username      TEXT UNIQUE,
            password_hash TEXT,
            created_at    REAL
        )
    """)

    # STT Lab — self-serve guest accounts (entirely separate from `managers`
    # and from client apps' API-key-mediated end users). A guest registers
    # directly on this service with no host app in the middle, so these
    # credentials only ever unlock that guest's own stt_adapters rows.
    c.execute("""
        CREATE TABLE IF NOT EXISTS stt_guests (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            username      TEXT UNIQUE,
            password_hash TEXT,
            created_at    REAL
        )
    """)

    # Tier 1 (hotword/prompt-bias) adapters. status/error_message/progress_message/
    # adapter_path are intentionally absent — Tier 1 creation is synchronous with
    # no training job. Tier 2 (LoRA fine-tune) will ALTER TABLE to add those,
    # same migration style as clients.webhook_url / voice_profiles.progress_message.
    c.execute("""
        CREATE TABLE IF NOT EXISTS stt_adapters (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            guest_id      INTEGER,
            name          TEXT,
            base_model    TEXT DEFAULT 'whisper-small',
            hotwords_json TEXT DEFAULT '[]',
            created_at    REAL
        )
    """)

    conn.commit()

    # Seed built-in system voices — per-voice check (not a one-time count==0 gate) so
    # a newly added BUILTIN_VOICES entry still gets seeded into an already-initialized DB.
    existing = {row[0] for row in c.execute(
        "SELECT base_tts_voice FROM voice_profiles WHERE kind='builtin'"
    ).fetchall()}
    for name, tts_voice in BUILTIN_VOICES:
        if tts_voice in existing:
            continue
        c.execute(
            "INSERT INTO voice_profiles (client_id, external_user_id, name, kind, base_tts_voice, status, created_at) "
            "VALUES (NULL, NULL, ?, 'builtin', ?, 'ready', ?)",
            (name, tts_voice, time.time())
        )
    conn.commit()

    # Seed the default client this service was split out of, on first run only.
    c.execute("SELECT id, api_key FROM clients WHERE name=?", (DEFAULT_CLIENT_NAME,))
    row = c.fetchone()
    if not row:
        api_key = secrets.token_urlsafe(32)
        c.execute(
            "INSERT INTO clients (name, api_key, created_at) VALUES (?,?,?)",
            (DEFAULT_CLIENT_NAME, api_key, time.time())
        )
        conn.commit()
        print(f"[clone-voice-station] Seeded client '{DEFAULT_CLIENT_NAME}' — API key: {api_key}")
        print(f"[clone-voice-station] Put this key in {DEFAULT_CLIENT_NAME}/voice_station_key.txt")

    # Seed a default manager dashboard account, on first run only, with a
    # random password printed once — there is no recovery flow yet, so write
    # it down (or change it from the dashboard after logging in).
    c.execute("SELECT COUNT(*) FROM managers")
    if c.fetchone()[0] == 0:
        password = secrets.token_urlsafe(12)
        c.execute(
            "INSERT INTO managers (username, password_hash, created_at) VALUES (?,?,?)",
            (DEFAULT_MANAGER_USERNAME, _hash_password(password), time.time())
        )
        conn.commit()
        print(f"[clone-voice-station] Seeded manager account — username: {DEFAULT_MANAGER_USERNAME}  password: {password}")
        print(f"[clone-voice-station] Log in at http://127.0.0.1:8090/login and change this password.")

    conn.close()


# =========================
# CLIENTS
# =========================
def get_client_by_api_key(api_key: str):
    if not api_key:
        return None
    conn = sqlite3.connect(DB_NAME)
    row  = conn.execute("SELECT id, name FROM clients WHERE api_key=?", (api_key,)).fetchone()
    conn.close()
    return {"id": row[0], "name": row[1]} if row else None


def list_clients():
    """Manager-dashboard view: every registered client app and its API key."""
    conn = sqlite3.connect(DB_NAME)
    rows = conn.execute(
        "SELECT id, name, api_key, webhook_url, created_at FROM clients ORDER BY id ASC"
    ).fetchall()
    conn.close()
    return [{"id": r[0], "name": r[1], "api_key": r[2], "webhook_url": r[3], "created_at": r[4]} for r in rows]


def create_client(name: str) -> dict:
    """Registers a new client app (e.g. a new corporate integration) with a
    freshly generated API key. Raises ValueError if the name is already taken."""
    conn = sqlite3.connect(DB_NAME)
    c    = conn.cursor()
    if c.execute("SELECT 1 FROM clients WHERE name=?", (name,)).fetchone():
        conn.close()
        raise ValueError(f"Client '{name}' đã tồn tại.")

    api_key = secrets.token_urlsafe(32)
    c.execute(
        "INSERT INTO clients (name, api_key, created_at) VALUES (?,?,?)",
        (name, api_key, time.time())
    )
    client_id = c.lastrowid
    conn.commit()
    conn.close()
    return {"id": client_id, "name": name, "api_key": api_key}


def delete_client(client_id: int):
    """Refuses to delete a client that still has voice profiles registered —
    delete/reassign those first so their data isn't silently orphaned."""
    conn = sqlite3.connect(DB_NAME)
    in_use = conn.execute(
        "SELECT COUNT(*) FROM voice_profiles WHERE client_id=?", (client_id,)
    ).fetchone()[0]
    if in_use:
        conn.close()
        raise ValueError(f"Client này còn {in_use} giọng nói — xoá hết trước khi xoá client.")
    conn.execute("DELETE FROM clients WHERE id=?", (client_id,))
    conn.commit()
    conn.close()


def get_client(client_id: int):
    conn = sqlite3.connect(DB_NAME)
    row  = conn.execute(
        "SELECT id, name, api_key, webhook_url, created_at FROM clients WHERE id=?", (client_id,)
    ).fetchone()
    conn.close()
    if not row:
        return None
    return {"id": row[0], "name": row[1], "api_key": row[2], "webhook_url": row[3], "created_at": row[4]}


def set_client_webhook(client_id: int, webhook_url: str):
    conn = sqlite3.connect(DB_NAME)
    conn.execute("UPDATE clients SET webhook_url=? WHERE id=?", (webhook_url or None, client_id))
    conn.commit()
    conn.close()


# =========================
# MANAGERS (dashboard login — separate from client apps' own users)
# =========================
def verify_manager_login(username: str, password: str) -> bool:
    conn = sqlite3.connect(DB_NAME)
    row  = conn.execute("SELECT password_hash FROM managers WHERE username=?", (username,)).fetchone()
    conn.close()
    return bool(row and _verify_password(password, row[0]))


def change_manager_password(username: str, new_password: str):
    conn = sqlite3.connect(DB_NAME)
    conn.execute(
        "UPDATE managers SET password_hash=? WHERE username=?",
        (_hash_password(new_password), username)
    )
    conn.commit()
    conn.close()


# =========================
# STT LAB (guest self-serve hotword/prompt-bias adapters — Tier 1)
# =========================
def create_stt_guest(username: str, password: str) -> int:
    """Raises sqlite3.IntegrityError if username is already taken — the
    caller (app.py) turns that into a 409."""
    conn = sqlite3.connect(DB_NAME)
    c    = conn.cursor()
    c.execute(
        "INSERT INTO stt_guests (username, password_hash, created_at) VALUES (?,?,?)",
        (username, _hash_password(password), time.time())
    )
    guest_id = c.lastrowid
    conn.commit()
    conn.close()
    return guest_id


def verify_stt_guest_login(username: str, password: str):
    conn = sqlite3.connect(DB_NAME)
    row  = conn.execute(
        "SELECT id, password_hash FROM stt_guests WHERE username=?", (username,)
    ).fetchone()
    conn.close()
    if not row or not _verify_password(password, row[1]):
        return None
    return {"id": row[0], "username": username}


def _stt_adapter_row_to_dict(row) -> dict:
    return {
        "id": row[0], "guest_id": row[1], "name": row[2], "base_model": row[3],
        "hotwords": json.loads(row[4] or "[]"), "created_at": row[5],
    }


def list_stt_adapters(guest_id: int) -> list:
    conn = sqlite3.connect(DB_NAME)
    rows = conn.execute(
        "SELECT id, guest_id, name, base_model, hotwords_json, created_at "
        "FROM stt_adapters WHERE guest_id=? ORDER BY id ASC",
        (guest_id,)
    ).fetchall()
    conn.close()
    return [_stt_adapter_row_to_dict(r) for r in rows]


def get_stt_adapter(adapter_id: int):
    conn = sqlite3.connect(DB_NAME)
    row  = conn.execute(
        "SELECT id, guest_id, name, base_model, hotwords_json, created_at "
        "FROM stt_adapters WHERE id=?",
        (adapter_id,)
    ).fetchone()
    conn.close()
    return _stt_adapter_row_to_dict(row) if row else None


def create_stt_adapter(guest_id: int, name: str) -> int:
    conn = sqlite3.connect(DB_NAME)
    c    = conn.cursor()
    c.execute(
        "INSERT INTO stt_adapters (guest_id, name, created_at) VALUES (?,?,?)",
        (guest_id, name, time.time())
    )
    adapter_id = c.lastrowid
    conn.commit()
    conn.close()
    return adapter_id


def rename_stt_adapter(adapter_id: int, name: str):
    conn = sqlite3.connect(DB_NAME)
    conn.execute("UPDATE stt_adapters SET name=? WHERE id=?", (name, adapter_id))
    conn.commit()
    conn.close()


def update_stt_adapter_hotwords(adapter_id: int, hotwords: list):
    conn = sqlite3.connect(DB_NAME)
    conn.execute(
        "UPDATE stt_adapters SET hotwords_json=? WHERE id=?",
        (json.dumps(hotwords), adapter_id)
    )
    conn.commit()
    conn.close()


def delete_stt_adapter(adapter_id: int):
    conn = sqlite3.connect(DB_NAME)
    conn.execute("DELETE FROM stt_adapters WHERE id=?", (adapter_id,))
    conn.commit()
    conn.close()


def delete_stt_guest_account(guest_id: int):
    """Full self-serve purge: every adapter this guest owns, then the account
    itself — honors the commitment that a guest's data is theirs to erase
    completely, not just disable."""
    conn = sqlite3.connect(DB_NAME)
    conn.execute("DELETE FROM stt_adapters WHERE guest_id=?", (guest_id,))
    conn.execute("DELETE FROM stt_guests WHERE id=?", (guest_id,))
    conn.commit()
    conn.close()


# =========================
# NOTIFICATIONS (manager-triggered delete/disable events → end user)
# =========================
def create_notification(client_id: int, external_user_id: str, profile_id: int,
                         profile_name: str, event: str, message: str) -> int:
    conn = sqlite3.connect(DB_NAME)
    c    = conn.cursor()
    c.execute(
        "INSERT INTO notifications (client_id, external_user_id, profile_id, profile_name, "
        "event, message, created_at, delivered_at) VALUES (?,?,?,?,?,?,?,NULL)",
        (client_id, external_user_id, profile_id, profile_name, event, message, time.time())
    )
    notification_id = c.lastrowid
    conn.commit()
    conn.close()
    return notification_id


def mark_notification_delivered(notification_id: int, client_id: int = None):
    """client_id, when given, scopes the update so one client can't ack another
    client's notification by guessing an id."""
    conn = sqlite3.connect(DB_NAME)
    if client_id is None:
        conn.execute("UPDATE notifications SET delivered_at=? WHERE id=?", (time.time(), notification_id))
    else:
        conn.execute(
            "UPDATE notifications SET delivered_at=? WHERE id=? AND client_id=?",
            (time.time(), notification_id, client_id)
        )
    conn.commit()
    conn.close()


def list_undelivered_notifications(client_id: int, external_user_id: str):
    """Polling fallback for a client app whose webhook is unset or was
    unreachable when the event fired."""
    conn = sqlite3.connect(DB_NAME)
    rows = conn.execute(
        "SELECT id, profile_id, profile_name, event, message, created_at FROM notifications "
        "WHERE client_id=? AND external_user_id=? AND delivered_at IS NULL ORDER BY created_at ASC",
        (client_id, external_user_id)
    ).fetchall()
    conn.close()
    return [
        {"id": r[0], "profile_id": r[1], "profile_name": r[2], "event": r[3], "message": r[4], "created_at": r[5]}
        for r in rows
    ]


# =========================
# SETTINGS (key/value store)
# =========================
def get_setting(key: str, default: str = "") -> str:
    conn = sqlite3.connect(DB_NAME)
    row  = conn.execute("SELECT value FROM settings WHERE key=?", (key,)).fetchone()
    conn.close()
    return row[0] if row and row[0] is not None else default


def set_setting(key: str, value: str):
    conn = sqlite3.connect(DB_NAME)
    c    = conn.cursor()
    c.execute(
        "INSERT INTO settings (key, value) VALUES (?,?) "
        "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
        (key, value)
    )
    conn.commit()
    conn.close()


# =========================
# VOICE CONSENT (disclaimer)
# =========================
def has_voice_consent(client_id: int, external_user_id: str) -> bool:
    conn = sqlite3.connect(DB_NAME)
    row  = conn.execute(
        "SELECT consented_at FROM voice_consent WHERE client_id=? AND external_user_id=?",
        (client_id, external_user_id)
    ).fetchone()
    conn.close()
    return bool(row and row[0])


def record_voice_consent(client_id: int, external_user_id: str):
    conn = sqlite3.connect(DB_NAME)
    c    = conn.cursor()
    c.execute(
        "INSERT INTO voice_consent (client_id, external_user_id, consented_at) VALUES (?,?,?) "
        "ON CONFLICT(client_id, external_user_id) DO UPDATE SET consented_at=excluded.consented_at",
        (client_id, external_user_id, time.time())
    )
    conn.commit()
    conn.close()


# =========================
# VOICE PROFILES
# =========================
def list_voice_profiles(client_id: int, external_user_id: str):
    """Built-in system voices + this end user's own cloned voices."""
    conn = sqlite3.connect(DB_NAME)
    c    = conn.cursor()
    c.execute(
        "SELECT id, client_id, external_user_id, name, kind, base_tts_voice, speaker_id, "
        "status, is_default, error_message, model_local_path "
        "FROM voice_profiles WHERE kind='builtin' OR (client_id=? AND external_user_id=?) "
        "ORDER BY kind DESC, id ASC",
        (client_id, external_user_id)
    )
    rows = c.fetchall()
    conn.close()
    return [_voice_profile_row_to_dict(r) for r in rows]


def _voice_profile_row_to_dict(r):
    return {
        "id":               r[0],
        "client_id":        r[1],
        "external_user_id": r[2],
        "name":             r[3],
        "kind":             r[4],
        "base_tts_voice":   r[5],
        "speaker_id":       r[6],
        "status":           r[7],
        "is_default":       bool(r[8]),
        "error_message":    r[9],
        "model_local_path": r[10],
    }


def get_voice_profile(profile_id: int):
    conn = sqlite3.connect(DB_NAME)
    row  = conn.execute(
        "SELECT id, client_id, external_user_id, name, kind, base_tts_voice, speaker_id, "
        "status, is_default, error_message, model_local_path "
        "FROM voice_profiles WHERE id=?",
        (profile_id,)
    ).fetchone()
    conn.close()
    return _voice_profile_row_to_dict(row) if row else None


def count_cloned_voice_profiles(client_id: int, external_user_id: str) -> int:
    conn = sqlite3.connect(DB_NAME)
    row  = conn.execute(
        "SELECT COUNT(*) FROM voice_profiles WHERE kind='cloned' AND client_id=? AND external_user_id=?",
        (client_id, external_user_id)
    ).fetchone()
    conn.close()
    return row[0]


def create_voice_profile(client_id: int, external_user_id: str, name: str, base_tts_voice: str = None) -> int:
    """Creates a new 'cloned' voice profile (starts empty, status='new')."""
    conn = sqlite3.connect(DB_NAME)
    c    = conn.cursor()
    c.execute(
        "INSERT INTO voice_profiles (client_id, external_user_id, name, kind, base_tts_voice, status, created_at) "
        "VALUES (?,?,?, 'cloned', ?, 'new', ?)",
        (client_id, external_user_id, name, base_tts_voice or BUILTIN_VOICES[0][1], time.time())
    )
    profile_id = c.lastrowid
    conn.commit()
    conn.close()
    return profile_id


def rename_voice_profile(profile_id: int, name: str):
    conn = sqlite3.connect(DB_NAME)
    c    = conn.cursor()
    c.execute("UPDATE voice_profiles SET name=? WHERE id=?", (name, profile_id))
    conn.commit()
    conn.close()


def set_default_voice_profile(client_id: int, external_user_id: str, profile_id: int):
    """Unsets any previous default for this end user, then sets the given profile as default.
    A built-in voice can also be set default per-user via a synthetic row lookup — callers
    just pass the profile id shown in list_voice_profiles(client_id, external_user_id)."""
    conn = sqlite3.connect(DB_NAME)
    c    = conn.cursor()
    c.execute(
        "UPDATE voice_profiles SET is_default=0 "
        "WHERE is_default=1 AND ((client_id=? AND external_user_id=?) OR kind='builtin')",
        (client_id, external_user_id)
    )
    c.execute("UPDATE voice_profiles SET is_default=1 WHERE id=?", (profile_id,))
    conn.commit()
    conn.close()


def update_voice_profile_status(profile_id: int, status: str, speaker_id: str = None,
                                 error_message: str = None, model_local_path: str = None):
    conn = sqlite3.connect(DB_NAME)
    c    = conn.cursor()

    fields = ["status=?"]
    params = [status]
    if speaker_id is not None:
        fields.append("speaker_id=?")
        params.append(speaker_id)
    if model_local_path is not None:
        fields.append("model_local_path=?")
        params.append(model_local_path)
    fields.append("error_message=?")
    params.append(error_message)
    params.append(profile_id)

    c.execute(f"UPDATE voice_profiles SET {', '.join(fields)} WHERE id=?", params)
    conn.commit()
    conn.close()


def delete_voice_profile(profile_id: int):
    """Deletes a cloned voice profile, its samples rows, sample files on disk,
    and the locally-backed-up trained model (voice_storage/<speaker_id>/), if any."""
    conn = sqlite3.connect(DB_NAME)
    c    = conn.cursor()
    row = c.execute("SELECT model_local_path FROM voice_profiles WHERE id=?", (profile_id,)).fetchone()
    model_dir = row[0] if row else None

    c.execute("DELETE FROM voice_samples WHERE profile_id=?", (profile_id,))
    c.execute("DELETE FROM voice_profiles WHERE id=? AND kind='cloned'", (profile_id,))
    conn.commit()
    conn.close()

    sample_dir = os.path.join(VOICE_SAMPLES_DIR, str(profile_id))
    if os.path.isdir(sample_dir):
        shutil.rmtree(sample_dir, ignore_errors=True)

    if model_dir and os.path.isdir(model_dir):
        shutil.rmtree(model_dir, ignore_errors=True)


def list_all_voice_profiles(client_id: int):
    """Admin view: every cloned voice profile belonging to this client, across all its
    end users."""
    conn = sqlite3.connect(DB_NAME)
    c    = conn.cursor()
    c.execute("""
        SELECT vp.id, vp.client_id, vp.external_user_id, vp.name, vp.kind, vp.base_tts_voice,
               vp.speaker_id, vp.status, vp.is_default, vp.error_message, vp.model_local_path,
               (SELECT COUNT(*) FROM voice_samples vs WHERE vs.profile_id = vp.id)
        FROM voice_profiles vp
        WHERE vp.kind='cloned' AND vp.client_id=?
        ORDER BY vp.id DESC
    """, (client_id,))
    rows = c.fetchall()
    conn.close()
    result = []
    for r in rows:
        d = _voice_profile_row_to_dict(r[:11])
        d["sample_count"] = r[11]
        result.append(d)
    return result


def list_all_voice_profiles_global():
    """Manager-dashboard view: every cloned voice profile across every client
    app, with the owning client's name attached."""
    conn = sqlite3.connect(DB_NAME)
    c    = conn.cursor()
    c.execute("""
        SELECT vp.id, vp.client_id, vp.external_user_id, vp.name, vp.kind, vp.base_tts_voice,
               vp.speaker_id, vp.status, vp.is_default, vp.error_message, vp.model_local_path,
               (SELECT COUNT(*) FROM voice_samples vs WHERE vs.profile_id = vp.id),
               cl.name
        FROM voice_profiles vp
        LEFT JOIN clients cl ON cl.id = vp.client_id
        WHERE vp.kind='cloned'
        ORDER BY vp.id DESC
    """)
    rows = c.fetchall()
    conn.close()
    result = []
    for r in rows:
        d = _voice_profile_row_to_dict(r[:11])
        d["sample_count"] = r[11]
        d["client_name"]  = r[12] or "(đã xoá)"
        result.append(d)
    return result


# =========================
# VOICE SAMPLES
# =========================
def add_voice_sample(profile_id: int, script_id: str, file_path: str) -> int:
    conn = sqlite3.connect(DB_NAME)
    c    = conn.cursor()
    c.execute(
        "INSERT INTO voice_samples (profile_id, script_id, file_path, created_at) VALUES (?,?,?,?)",
        (profile_id, script_id, file_path, time.time())
    )
    sample_id = c.lastrowid
    # First sample moves a fresh profile into 'collecting'
    c.execute(
        "UPDATE voice_profiles SET status='collecting' WHERE id=? AND status='new'",
        (profile_id,)
    )
    conn.commit()
    conn.close()
    return sample_id


def list_voice_samples(profile_id: int):
    conn = sqlite3.connect(DB_NAME)
    c    = conn.cursor()
    c.execute(
        "SELECT id, script_id, file_path, created_at FROM voice_samples WHERE profile_id=? ORDER BY created_at ASC",
        (profile_id,)
    )
    rows = c.fetchall()
    conn.close()
    return [{"id": r[0], "script_id": r[1], "file_path": r[2], "created_at": r[3]} for r in rows]


def delete_voice_sample(sample_id: int):
    conn = sqlite3.connect(DB_NAME)
    c    = conn.cursor()
    row  = c.execute("SELECT file_path FROM voice_samples WHERE id=?", (sample_id,)).fetchone()
    c.execute("DELETE FROM voice_samples WHERE id=?", (sample_id,))
    conn.commit()
    conn.close()
    if row and row[0] and os.path.exists(row[0]):
        try:
            os.remove(row[0])
        except OSError:
            pass
