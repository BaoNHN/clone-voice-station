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

STT_SAMPLES_DIR = os.path.join(BASE_DIR, "stt_training_samples")
STT_MODELS_DIR  = os.path.join(BASE_DIR, "stt_adapters_storage")

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

# STT Lab Tier 2 (LoRA fine-tune) training-data bounds.
MIN_STT_TRAIN_SAMPLES = 10
MAX_STT_TRAIN_SAMPLES = 500
MAX_STT_SAMPLE_DURATION_SEC = 30
MAX_STT_ADAPTERS_PER_GUEST = 3
# First entry is the default base model.
ALLOWED_STT_BASE_MODELS = ("phowhisper-small", "whisper-tiny", "whisper-base")

# First client seeded on a fresh DB, matching the app this service was
# extracted from.
DEFAULT_CLIENT_NAME = "voice-rag-example"

# Default account for the manager dashboard, seeded on first init only.
DEFAULT_MANAGER_USERNAME = "manager"

PBKDF2_ITERATIONS = 200_000


def get_conn():
    # timeout=30 (up from sqlite3's 5s default) avoids "database is locked"
    # under concurrent writers; see init_db()'s WAL pragma for the other half.
    return sqlite3.connect(DB_NAME, timeout=30)


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
    conn = get_conn()
    # WAL mode persists in the DB file once set; lets readers proceed without
    # blocking on a writer.
    conn.execute("PRAGMA journal_mode=WAL")
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

    # Manager-triggered delete/disable events, delivered via webhook or polled
    # via GET /api/notifications while delivered_at is NULL.
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
    # Each guest gets a dedicated clients row (see _provision_stt_guest_client),
    # so they can self-publish without a manager assigning a client first.
    try:
        c.execute("ALTER TABLE stt_guests ADD COLUMN client_id INTEGER")
    except Exception:
        pass

    # Tier 1 (hotword/prompt-bias) adapters. status/error_message/progress_message/
    # adapter_path are intentionally absent — Tier 1 creation is synchronous with
    # no training job. Tier 2 (LoRA fine-tune) will ALTER TABLE to add those,
    # same migration style as clients.webhook_url / voice_profiles.progress_message.
    c.execute("""
        CREATE TABLE IF NOT EXISTS stt_adapters (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            guest_id      INTEGER,
            name          TEXT,
            base_model    TEXT DEFAULT 'whisper-tiny',
            hotwords_json TEXT DEFAULT '[]',
            created_at    REAL
        )
    """)
    # Tier 2 (LoRA fine-tune) migration — existing Tier-1-only rows default to
    # status='ready' since a hotword-only adapter is already usable as-is; they
    # only move through training/failed once a guest actually triggers /train.
    for stmt in (
        "ALTER TABLE stt_adapters ADD COLUMN status TEXT DEFAULT 'ready'",
        "ALTER TABLE stt_adapters ADD COLUMN error_message TEXT",
        "ALTER TABLE stt_adapters ADD COLUMN progress_message TEXT",
        "ALTER TABLE stt_adapters ADD COLUMN adapter_path TEXT",
        "ALTER TABLE stt_adapters ADD COLUMN resume_from_path TEXT",
        "ALTER TABLE stt_adapters ADD COLUMN backend_used TEXT",
        # client_id: which client's /api/transcribe should use this adapter.
        # is_published gates that, at most one per client_id (see publish_stt_adapter()).
        "ALTER TABLE stt_adapters ADD COLUMN client_id INTEGER",
        "ALTER TABLE stt_adapters ADD COLUMN is_published INTEGER DEFAULT 0",
        # System-wide fallback adapter for any client with none published; see
        # get_default_stt_adapter().
        "ALTER TABLE stt_adapters ADD COLUMN is_default INTEGER DEFAULT 0",
    ):
        try:
            c.execute(stmt)
        except Exception:
            pass

    # Tier 2 training data — (audio, transcript) pairs a guest uploads to fine-tune
    # their adapter. Separate from stt_adapters.hotwords_json (Tier 1, no audio).
    c.execute("""
        CREATE TABLE IF NOT EXISTS stt_training_samples (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            adapter_id     INTEGER,
            audio_path     TEXT,
            reference_text TEXT,
            created_at     REAL,
            is_holdout     INTEGER DEFAULT 0
        )
    """)
    # is_holdout marks a sample as part of a genuinely independent test set
    # (see voice/stt_local_train.py's train() holdout_samples param).
    try:
        c.execute("ALTER TABLE stt_training_samples ADD COLUMN is_holdout INTEGER DEFAULT 0")
    except Exception:
        pass

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

    # One-time rename from the old display-style default client name; keeps
    # the same id/api_key so already-configured clients keep working.
    c.execute("UPDATE clients SET name=? WHERE name=?", (DEFAULT_CLIENT_NAME, "Voice Rag example"))
    if c.rowcount:
        conn.commit()
        print(f"[clone-voice-station] Renamed default client 'Voice Rag example' -> '{DEFAULT_CLIENT_NAME}'")

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

    # Backfill client_id for guests registered before self-publish existed.
    orphan_guests = c.execute("SELECT id, username FROM stt_guests WHERE client_id IS NULL").fetchall()
    for guest_id, username in orphan_guests:
        try:
            client_id = _provision_stt_guest_client(username)
            c.execute("UPDATE stt_guests SET client_id=? WHERE id=?", (client_id, guest_id))
            conn.commit()
            print(f"[clone-voice-station] Backfilled client_id for STT guest '{username}' (id={guest_id})")
        except Exception as e:
            print(f"[clone-voice-station] Failed to backfill client for STT guest '{username}' (id={guest_id}): {e}")

    # Recover training rows orphaned by a server crash/restart: flips any row
    # stuck at status='training' to 'error' so its retrain button unlocks
    # again. Names printed here stay ASCII-only (Windows console encoding).
    orphaned_profiles = c.execute(
        "SELECT id FROM voice_profiles WHERE status='training'"
    ).fetchall()
    for (pid,) in orphaned_profiles:
        c.execute(
            "UPDATE voice_profiles SET status='error', error_message=? WHERE id=?",
            ("Huấn luyện bị gián đoạn (máy chủ khởi động lại giữa chừng) — bấm \"Huấn luyện lại\" để tiếp tục.", pid)
        )
        print(f"[clone-voice-station] Recovered orphaned training for voice profile id={pid} -> error, retrain unlocked")
    if orphaned_profiles:
        conn.commit()

    orphaned_adapters = c.execute(
        "SELECT id FROM stt_adapters WHERE status='training'"
    ).fetchall()
    for (aid,) in orphaned_adapters:
        c.execute(
            "UPDATE stt_adapters SET status='error', error_message=? WHERE id=?",
            ("Huấn luyện bị gián đoạn (máy chủ khởi động lại giữa chừng) — bấm \"Huấn luyện lại\" để tiếp tục.", aid)
        )
        print(f"[clone-voice-station] Recovered orphaned training for STT adapter id={aid} -> error, retrain unlocked")
    if orphaned_adapters:
        conn.commit()

    conn.close()


# =========================
# CLIENTS
# =========================
def get_client_by_api_key(api_key: str):
    if not api_key:
        return None
    conn = get_conn()
    row  = conn.execute("SELECT id, name FROM clients WHERE api_key=?", (api_key,)).fetchone()
    conn.close()
    return {"id": row[0], "name": row[1]} if row else None


def list_clients():
    """Manager-dashboard view: every registered client app and its API key."""
    conn = get_conn()
    rows = conn.execute(
        "SELECT id, name, api_key, webhook_url, created_at FROM clients ORDER BY id ASC"
    ).fetchall()
    conn.close()
    return [{"id": r[0], "name": r[1], "api_key": r[2], "webhook_url": r[3], "created_at": r[4]} for r in rows]


def create_client(name: str) -> dict:
    """Registers a new client app (e.g. a new corporate integration) with a
    freshly generated API key. Raises ValueError if the name is already taken."""
    conn = get_conn()
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
    conn = get_conn()
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
    conn = get_conn()
    row  = conn.execute(
        "SELECT id, name, api_key, webhook_url, created_at FROM clients WHERE id=?", (client_id,)
    ).fetchone()
    conn.close()
    if not row:
        return None
    return {"id": row[0], "name": row[1], "api_key": row[2], "webhook_url": row[3], "created_at": row[4]}


def set_client_webhook(client_id: int, webhook_url: str):
    conn = get_conn()
    conn.execute("UPDATE clients SET webhook_url=? WHERE id=?", (webhook_url or None, client_id))
    conn.commit()
    conn.close()


# =========================
# MANAGERS (dashboard login — separate from client apps' own users)
# =========================
def verify_manager_login(username: str, password: str) -> bool:
    conn = get_conn()
    row  = conn.execute("SELECT password_hash FROM managers WHERE username=?", (username,)).fetchone()
    conn.close()
    return bool(row and _verify_password(password, row[0]))


def change_manager_password(username: str, new_password: str):
    conn = get_conn()
    conn.execute(
        "UPDATE managers SET password_hash=? WHERE username=?",
        (_hash_password(new_password), username)
    )
    conn.commit()
    conn.close()


# =========================
# STT LAB (guest self-serve hotword/prompt-bias adapters — Tier 1)
# =========================
def _provision_stt_guest_client(username: str) -> int:
    """Auto-provisions a dedicated `clients` row for an STT Lab guest so they can
    self-publish adapters (see publish_stt_adapter_route in app.py) without a manager
    assigning one -- isolated per guest, never a real production client like
    DEFAULT_CLIENT_NAME. `username` is UNIQUE on stt_guests, so this name only ever
    collides with create_client()'s uniqueness check if a client with that exact name
    was created some other way; fall back to a suffixed name in that rare case rather
    than fail registration outright."""
    name = f"stt-guest-{username}"
    try:
        client = create_client(name)
    except ValueError:
        client = create_client(f"{name}-{secrets.token_hex(4)}")
    return client["id"]


def create_stt_guest(username: str, password: str) -> int:
    """Raises sqlite3.IntegrityError if username is already taken (app.py
    turns that into a 409). try/finally ensures the connection always closes
    even on that error, avoiding a leaked open transaction that would
    otherwise block every other write to this DB."""
    conn = get_conn()
    try:
        c = conn.cursor()
        c.execute(
            "INSERT INTO stt_guests (username, password_hash, created_at) VALUES (?,?,?)",
            (username, _hash_password(password), time.time())
        )
        guest_id = c.lastrowid
        conn.commit()
    finally:
        conn.close()

    # Provisioned as a separate step/connection (not inside the transaction above) --
    # if this fails, the guest account still exists and init_db()'s startup backfill
    # will retry it, rather than failing registration outright over a client-side add.
    try:
        client_id = _provision_stt_guest_client(username)
        conn = get_conn()
        conn.execute("UPDATE stt_guests SET client_id=? WHERE id=?", (client_id, guest_id))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"[clone-voice-station] Failed to provision client for new STT guest '{username}': {e}")

    return guest_id


def get_stt_guest(guest_id: int):
    conn = get_conn()
    row  = conn.execute(
        "SELECT id, username, client_id FROM stt_guests WHERE id=?", (guest_id,)
    ).fetchone()
    conn.close()
    if not row:
        return None
    return {"id": row[0], "username": row[1], "client_id": row[2]}


def verify_stt_guest_login(username: str, password: str):
    conn = get_conn()
    row  = conn.execute(
        "SELECT id, password_hash FROM stt_guests WHERE username=?", (username,)
    ).fetchone()
    conn.close()
    if not row or not _verify_password(password, row[1]):
        return None
    return {"id": row[0], "username": username}


_STT_ADAPTER_COLUMNS = (
    "id, guest_id, name, base_model, hotwords_json, created_at, "
    "status, error_message, progress_message, adapter_path, resume_from_path, backend_used, "
    "client_id, is_published, is_default"
)


def _stt_adapter_row_to_dict(row) -> dict:
    return {
        "id": row[0], "guest_id": row[1], "name": row[2], "base_model": row[3],
        "hotwords": json.loads(row[4] or "[]"), "created_at": row[5],
        "status": row[6], "error_message": row[7], "progress_message": row[8],
        "adapter_path": row[9], "resume_from_path": row[10], "backend_used": row[11],
        "client_id": row[12], "is_published": bool(row[13]), "is_default": bool(row[14]),
    }


def list_stt_adapters(guest_id: int) -> list:
    conn = get_conn()
    rows = conn.execute(
        f"SELECT {_STT_ADAPTER_COLUMNS} FROM stt_adapters WHERE guest_id=? ORDER BY id ASC",
        (guest_id,)
    ).fetchall()
    conn.close()
    return [_stt_adapter_row_to_dict(r) for r in rows]


def get_stt_adapter(adapter_id: int):
    conn = get_conn()
    row  = conn.execute(
        f"SELECT {_STT_ADAPTER_COLUMNS} FROM stt_adapters WHERE id=?",
        (adapter_id,)
    ).fetchone()
    conn.close()
    return _stt_adapter_row_to_dict(row) if row else None


def count_stt_adapters(guest_id: int) -> int:
    conn = get_conn()
    n = conn.execute("SELECT COUNT(*) FROM stt_adapters WHERE guest_id=?", (guest_id,)).fetchone()[0]
    conn.close()
    return n


def create_stt_adapter(guest_id: int, name: str, base_model: str = "whisper-tiny") -> int:
    """guest_id=None for a manager-created adapter (see app.py's
    /manager/stt/adapters routes) -- it has no self-serve owner, only ever
    reachable/manageable by a manager, until published to a client."""
    conn = get_conn()
    c    = conn.cursor()
    c.execute(
        "INSERT INTO stt_adapters (guest_id, name, base_model, status, created_at) VALUES (?,?,?,?,?)",
        (guest_id, name, base_model, "ready", time.time())
    )
    adapter_id = c.lastrowid
    conn.commit()
    conn.close()
    return adapter_id


def list_all_stt_adapters_global() -> list:
    """Manager-dashboard view: every STT Tier 2 adapter across both guests and
    manager-created ones, with the owning guest's username (if any) and the
    published-to client's name (if any) attached -- mirrors
    list_all_voice_profiles_global()'s pattern for RVC voices."""
    conn = get_conn()
    rows = conn.execute(f"""
        SELECT {', '.join('sa.' + col.strip() for col in _STT_ADAPTER_COLUMNS.split(','))},
               sg.username, cl.name
        FROM stt_adapters sa
        LEFT JOIN stt_guests sg ON sg.id = sa.guest_id
        LEFT JOIN clients cl ON cl.id = sa.client_id
        ORDER BY sa.id DESC
    """).fetchall()
    conn.close()
    result = []
    for r in rows:
        d = _stt_adapter_row_to_dict(r[:15])
        d["guest_username"] = r[15]
        d["client_name"] = r[16]
        result.append(d)
    return result


def publish_stt_adapter(adapter_id: int, client_id: int):
    """Marks adapter_id as the active Tier 2 model for client_id's /api/transcribe
    calls -- unpublishes any adapter previously published for that same client
    first, so at most one is ever active per client (same pattern as
    voice_profiles.is_default's single-default-per-user invariant)."""
    conn = get_conn()
    conn.execute("UPDATE stt_adapters SET is_published=0 WHERE client_id=? AND is_published=1", (client_id,))
    conn.execute("UPDATE stt_adapters SET client_id=?, is_published=1 WHERE id=?", (client_id, adapter_id))
    conn.commit()
    conn.close()


def unpublish_stt_adapter(adapter_id: int):
    conn = get_conn()
    conn.execute("UPDATE stt_adapters SET is_published=0 WHERE id=?", (adapter_id,))
    conn.commit()
    conn.close()


def set_default_stt_adapter(adapter_id: int):
    """Designates adapter_id as the system-wide fallback STT model used by any
    client with no adapter of its own published (see get_default_stt_adapter()
    and /api/transcribe's fallback chain) -- independent of client_id/
    is_published, which only ever target one specific client. At most one
    default at a time, same single-row invariant as is_published-per-client."""
    conn = get_conn()
    conn.execute("UPDATE stt_adapters SET is_default=0 WHERE is_default=1")
    conn.execute("UPDATE stt_adapters SET is_default=1 WHERE id=?", (adapter_id,))
    conn.commit()
    conn.close()


def unset_default_stt_adapter(adapter_id: int):
    conn = get_conn()
    conn.execute("UPDATE stt_adapters SET is_default=0 WHERE id=?", (adapter_id,))
    conn.commit()
    conn.close()


def get_default_stt_adapter():
    """Only ever returns a default that's both flagged AND actually ready to run
    -- same staleness guard as get_published_stt_adapter_for_client()."""
    conn = get_conn()
    row = conn.execute(
        f"SELECT {_STT_ADAPTER_COLUMNS} FROM stt_adapters "
        f"WHERE is_default=1 AND status='ready' AND adapter_path IS NOT NULL LIMIT 1"
    ).fetchone()
    conn.close()
    return _stt_adapter_row_to_dict(row) if row else None


def get_published_stt_adapter_for_client(client_id: int):
    """Used by /api/transcribe to decide whether to route through a fine-tuned
    adapter instead of the base PhoWhisper/Whisper path -- only ever returns an
    adapter that is both published AND has actually finished training
    successfully (status='ready' with a real adapter_path), never a stale
    'published' flag left over from before a retrain that hasn't completed or
    that failed."""
    conn = get_conn()
    row = conn.execute(
        f"SELECT {_STT_ADAPTER_COLUMNS} FROM stt_adapters "
        f"WHERE client_id=? AND is_published=1 AND status='ready' AND adapter_path IS NOT NULL "
        f"LIMIT 1",
        (client_id,)
    ).fetchone()
    conn.close()
    return _stt_adapter_row_to_dict(row) if row else None


def rename_stt_adapter(adapter_id: int, name: str):
    conn = get_conn()
    conn.execute("UPDATE stt_adapters SET name=? WHERE id=?", (name, adapter_id))
    conn.commit()
    conn.close()


def update_stt_adapter_hotwords(adapter_id: int, hotwords: list):
    conn = get_conn()
    conn.execute(
        "UPDATE stt_adapters SET hotwords_json=? WHERE id=?",
        (json.dumps(hotwords), adapter_id)
    )
    conn.commit()
    conn.close()


def update_stt_adapter_training(adapter_id: int, status: str, error_message: str = None,
                                 progress_message: str = None, adapter_path: str = None,
                                 backend_used: str = None):
    """status/error_message/progress_message are always overwritten (each call
    represents the adapter's current state); adapter_path/backend_used only
    overwrite when actually provided, via COALESCE, so an in-progress poll
    update doesn't erase a previous successful training run's artifact path."""
    conn = get_conn()
    conn.execute(
        "UPDATE stt_adapters SET status=?, error_message=?, progress_message=?, "
        "adapter_path=COALESCE(?, adapter_path), backend_used=COALESCE(?, backend_used) WHERE id=?",
        (status, error_message, progress_message, adapter_path, backend_used, adapter_id)
    )
    conn.commit()
    conn.close()


def set_stt_adapter_resume_path(adapter_id: int, path: str):
    conn = get_conn()
    conn.execute("UPDATE stt_adapters SET resume_from_path=? WHERE id=?", (path, adapter_id))
    conn.commit()
    conn.close()


def add_stt_training_sample(adapter_id: int, audio_path: str, reference_text: str, is_holdout: bool = False) -> int:
    conn = get_conn()
    c    = conn.cursor()
    c.execute(
        "INSERT INTO stt_training_samples (adapter_id, audio_path, reference_text, created_at, is_holdout) VALUES (?,?,?,?,?)",
        (adapter_id, audio_path, reference_text, time.time(), int(is_holdout))
    )
    sample_id = c.lastrowid
    conn.commit()
    conn.close()
    return sample_id


def _stt_sample_row_to_dict(row) -> dict:
    return {
        "id": row[0], "adapter_id": row[1], "audio_path": row[2], "reference_text": row[3],
        "created_at": row[4], "is_holdout": bool(row[5]),
    }


def list_stt_training_samples(adapter_id: int) -> list:
    conn = get_conn()
    rows = conn.execute(
        "SELECT id, adapter_id, audio_path, reference_text, created_at, is_holdout "
        "FROM stt_training_samples WHERE adapter_id=? ORDER BY id ASC",
        (adapter_id,)
    ).fetchall()
    conn.close()
    return [_stt_sample_row_to_dict(r) for r in rows]


def get_stt_training_sample(sample_id: int):
    conn = get_conn()
    row = conn.execute(
        "SELECT id, adapter_id, audio_path, reference_text, created_at, is_holdout FROM stt_training_samples WHERE id=?",
        (sample_id,)
    ).fetchone()
    conn.close()
    return _stt_sample_row_to_dict(row) if row else None


def update_stt_training_sample_text(sample_id: int, reference_text: str):
    conn = get_conn()
    conn.execute("UPDATE stt_training_samples SET reference_text=? WHERE id=?", (reference_text, sample_id))
    conn.commit()
    conn.close()


def delete_stt_training_sample(sample_id: int):
    """Caller (app.py) removes the audio file from disk first — this only
    drops the DB row."""
    conn = get_conn()
    conn.execute("DELETE FROM stt_training_samples WHERE id=?", (sample_id,))
    conn.commit()
    conn.close()


def delete_stt_adapter(adapter_id: int):
    """Caller (app.py) removes sample audio files + the trained adapter
    directory from disk first — this only drops the DB rows (cascades
    stt_training_samples)."""
    conn = get_conn()
    conn.execute("DELETE FROM stt_training_samples WHERE adapter_id=?", (adapter_id,))
    conn.execute("DELETE FROM stt_adapters WHERE id=?", (adapter_id,))
    conn.commit()
    conn.close()


def delete_stt_guest_account(guest_id: int):
    """Full self-serve purge: every training sample + adapter this guest owns,
    then the account itself — honors the commitment that a guest's data is
    theirs to erase completely, not just disable. Caller (app.py) removes
    files from disk first (needs the paths before the rows disappear)."""
    conn = get_conn()
    conn.execute(
        "DELETE FROM stt_training_samples WHERE adapter_id IN "
        "(SELECT id FROM stt_adapters WHERE guest_id=?)",
        (guest_id,)
    )
    conn.execute("DELETE FROM stt_adapters WHERE guest_id=?", (guest_id,))
    conn.execute("DELETE FROM stt_guests WHERE id=?", (guest_id,))
    conn.commit()
    conn.close()


# =========================
# NOTIFICATIONS (manager-triggered delete/disable events → end user)
# =========================
def create_notification(client_id: int, external_user_id: str, profile_id: int,
                         profile_name: str, event: str, message: str) -> int:
    conn = get_conn()
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
    conn = get_conn()
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
    conn = get_conn()
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
    conn = get_conn()
    row  = conn.execute("SELECT value FROM settings WHERE key=?", (key,)).fetchone()
    conn.close()
    return row[0] if row and row[0] is not None else default


def set_setting(key: str, value: str):
    conn = get_conn()
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
    conn = get_conn()
    row  = conn.execute(
        "SELECT consented_at FROM voice_consent WHERE client_id=? AND external_user_id=?",
        (client_id, external_user_id)
    ).fetchone()
    conn.close()
    return bool(row and row[0])


def record_voice_consent(client_id: int, external_user_id: str):
    conn = get_conn()
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
    conn = get_conn()
    c    = conn.cursor()
    c.execute(
        "SELECT id, client_id, external_user_id, name, kind, base_tts_voice, speaker_id, "
        "status, is_default, error_message, model_local_path, progress_message "
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
        "progress_message": r[11],
    }


def get_voice_profile(profile_id: int):
    conn = get_conn()
    row  = conn.execute(
        "SELECT id, client_id, external_user_id, name, kind, base_tts_voice, speaker_id, "
        "status, is_default, error_message, model_local_path, progress_message "
        "FROM voice_profiles WHERE id=?",
        (profile_id,)
    ).fetchone()
    conn.close()
    return _voice_profile_row_to_dict(row) if row else None


def count_cloned_voice_profiles(client_id: int, external_user_id: str) -> int:
    conn = get_conn()
    row  = conn.execute(
        "SELECT COUNT(*) FROM voice_profiles WHERE kind='cloned' AND client_id=? AND external_user_id=?",
        (client_id, external_user_id)
    ).fetchone()
    conn.close()
    return row[0]


def create_voice_profile(client_id: int, external_user_id: str, name: str, base_tts_voice: str = None) -> int:
    """Creates a new 'cloned' voice profile (starts empty, status='new')."""
    conn = get_conn()
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
    conn = get_conn()
    c    = conn.cursor()
    c.execute("UPDATE voice_profiles SET name=? WHERE id=?", (name, profile_id))
    conn.commit()
    conn.close()


# Base voices a cloned profile may be built on. Restricted to the known list
# since the value is handed straight to edge-TTS.
VALID_BASE_TTS_VOICES = {voice_id for _, voice_id in BUILTIN_VOICES}


def set_voice_profile_base_voice(profile_id: int, base_tts_voice: str):
    """Changes which TTS voice a profile is synthesised from before RVC runs.

    Takes effect on the next /api/speak call; the trained RVC model itself is
    unaffected, so switching base voices needs no retraining.
    """
    if base_tts_voice not in VALID_BASE_TTS_VOICES:
        raise ValueError(f"unknown base_tts_voice: {base_tts_voice!r}")
    conn = get_conn()
    c    = conn.cursor()
    c.execute("UPDATE voice_profiles SET base_tts_voice=? WHERE id=?", (base_tts_voice, profile_id))
    conn.commit()
    conn.close()


def set_default_voice_profile(client_id: int, external_user_id: str, profile_id: int):
    """Unsets any previous default for this end user, then sets the given profile as default.
    A built-in voice can also be set default per-user via a synthetic row lookup — callers
    just pass the profile id shown in list_voice_profiles(client_id, external_user_id)."""
    conn = get_conn()
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
                                 error_message: str = None, model_local_path: str = None,
                                 progress_message: str = None):
    conn = get_conn()
    c    = conn.cursor()

    fields = ["status=?"]
    params = [status]
    if speaker_id is not None:
        fields.append("speaker_id=?")
        params.append(speaker_id)
    if model_local_path is not None:
        fields.append("model_local_path=?")
        params.append(model_local_path)
    # Unlike error_message (always overwritten -- a fresh status implies any old
    # error is stale), progress_message is only touched when the caller actually
    # has something new to report, since most status transitions (e.g. the final
    # "ready"/"failed") don't pass one and shouldn't blank out the last step shown.
    if progress_message is not None:
        fields.append("progress_message=?")
        params.append(progress_message)
    fields.append("error_message=?")
    params.append(error_message)
    params.append(profile_id)

    c.execute(f"UPDATE voice_profiles SET {', '.join(fields)} WHERE id=?", params)
    conn.commit()
    conn.close()


def delete_voice_profile(profile_id: int):
    """Deletes a cloned voice profile, its samples rows, sample files on disk,
    and the locally-backed-up trained model (voice_storage/<speaker_id>/), if any."""
    conn = get_conn()
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
    conn = get_conn()
    c    = conn.cursor()
    c.execute("""
        SELECT vp.id, vp.client_id, vp.external_user_id, vp.name, vp.kind, vp.base_tts_voice,
               vp.speaker_id, vp.status, vp.is_default, vp.error_message, vp.model_local_path,
               vp.progress_message,
               (SELECT COUNT(*) FROM voice_samples vs WHERE vs.profile_id = vp.id)
        FROM voice_profiles vp
        WHERE vp.kind='cloned' AND vp.client_id=?
        ORDER BY vp.id DESC
    """, (client_id,))
    rows = c.fetchall()
    conn.close()
    result = []
    for r in rows:
        d = _voice_profile_row_to_dict(r[:12])
        d["sample_count"] = r[12]
        result.append(d)
    return result


def list_all_voice_profiles_global():
    """Manager-dashboard view: every cloned voice profile across every client
    app, with the owning client's name attached."""
    conn = get_conn()
    c    = conn.cursor()
    c.execute("""
        SELECT vp.id, vp.client_id, vp.external_user_id, vp.name, vp.kind, vp.base_tts_voice,
               vp.speaker_id, vp.status, vp.is_default, vp.error_message, vp.model_local_path,
               vp.progress_message,
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
        d = _voice_profile_row_to_dict(r[:12])
        d["sample_count"] = r[12]
        d["client_name"]  = r[13] or "(đã xoá)"
        result.append(d)
    return result


# =========================
# VOICE SAMPLES
# =========================
def add_voice_sample(profile_id: int, script_id: str, file_path: str) -> int:
    conn = get_conn()
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
    conn = get_conn()
    c    = conn.cursor()
    c.execute(
        "SELECT id, script_id, file_path, created_at FROM voice_samples WHERE profile_id=? ORDER BY created_at ASC",
        (profile_id,)
    )
    rows = c.fetchall()
    conn.close()
    return [{"id": r[0], "script_id": r[1], "file_path": r[2], "created_at": r[3]} for r in rows]


def delete_voice_sample(sample_id: int):
    conn = get_conn()
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
