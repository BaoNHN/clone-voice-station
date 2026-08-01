"""
migrate_from_rag_legal_assistant.py
One-time migration: copies existing cloned voice profiles/samples/consent from
rag-legal-assistant's chat.db + voice_samples/ + voice_storage/ into this
service's own DB/storage, under the "rag-legal-assistant" client.

Usage:
    python migrate_from_rag_legal_assistant.py [path-to-rag-legal-assistant]

Defaults to the sibling directory ../rag-legal-assistant if no path is given.
Safe to re-run: already-migrated profiles are tracked in
migration_old_profile_map and skipped on subsequent runs.

Builtin voices are NOT migrated — this service seeds its own BUILTIN_VOICES
on first init_db() (see database/database.py), independent of the source app.
"""

import os
import secrets
import shutil
import sqlite3
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from database.database import init_db, DB_NAME, VOICE_SAMPLES_DIR, VOICE_MODELS_DIR, DEFAULT_CLIENT_NAME


def get_or_create_default_client(conn):
    c   = conn.cursor()
    row = c.execute("SELECT id, api_key FROM clients WHERE name=?", (DEFAULT_CLIENT_NAME,)).fetchone()
    if row:
        return row[0], row[1]
    api_key = secrets.token_urlsafe(32)
    c.execute(
        "INSERT INTO clients (name, api_key, created_at) VALUES (?,?,?)",
        (DEFAULT_CLIENT_NAME, api_key, time.time())
    )
    conn.commit()
    return c.lastrowid, api_key


def ensure_migration_table(conn):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS migration_old_profile_map (
            old_profile_id INTEGER PRIMARY KEY,
            new_profile_id INTEGER
        )
    """)
    conn.commit()


def main():
    src_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "rag-legal-assistant"
    )
    src_dir = os.path.abspath(src_dir)

    old_db_path      = os.path.join(src_dir, "chat.db")
    old_models_dir   = os.path.join(src_dir, "voice_storage")

    if not os.path.exists(old_db_path):
        print(f"[migrate] Old chat.db not found at {old_db_path}")
        sys.exit(1)

    init_db()  # ensure this service's own schema/tables (and builtin voices) exist

    new_conn = sqlite3.connect(DB_NAME)
    ensure_migration_table(new_conn)
    client_id, api_key = get_or_create_default_client(new_conn)

    old_conn = sqlite3.connect(old_db_path)

    # ── Voice profiles (cloned only) ──
    old_profiles = old_conn.execute(
        "SELECT id, user_id, name, kind, base_tts_voice, speaker_id, status, is_default, "
        "error_message, model_local_path, created_at FROM voice_profiles WHERE kind='cloned'"
    ).fetchall()

    id_map = {}
    migrated_profiles = 0
    for row in old_profiles:
        (old_id, user_id, name, kind, base_tts_voice, speaker_id, status,
         is_default, error_message, model_local_path, created_at) = row

        already = new_conn.execute(
            "SELECT new_profile_id FROM migration_old_profile_map WHERE old_profile_id=?", (old_id,)
        ).fetchone()
        if already:
            id_map[old_id] = already[0]
            continue

        external_user_id = str(user_id)
        c = new_conn.cursor()
        c.execute(
            "INSERT INTO voice_profiles (client_id, external_user_id, name, kind, base_tts_voice, "
            "speaker_id, status, is_default, error_message, model_local_path, created_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (client_id, external_user_id, name, kind, base_tts_voice, speaker_id, status,
             is_default, error_message, model_local_path, created_at)
        )
        new_id = c.lastrowid
        new_conn.execute(
            "INSERT INTO migration_old_profile_map (old_profile_id, new_profile_id) VALUES (?,?)",
            (old_id, new_id)
        )
        new_conn.commit()
        id_map[old_id] = new_id
        migrated_profiles += 1

        # Local trained-model backup (voice_storage/<speaker_id>/) — keyed by speaker_id,
        # unaffected by the profile-id remap.
        if model_local_path and speaker_id:
            old_model_dir = os.path.join(old_models_dir, speaker_id)
            new_model_dir = os.path.join(VOICE_MODELS_DIR, speaker_id)
            if os.path.isdir(old_model_dir) and not os.path.isdir(new_model_dir):
                shutil.copytree(old_model_dir, new_model_dir)

    # ── Voice samples ──
    migrated_samples = 0
    for old_id, new_id in id_map.items():
        old_samples = old_conn.execute(
            "SELECT script_id, file_path, created_at FROM voice_samples WHERE profile_id=?", (old_id,)
        ).fetchall()
        if not old_samples:
            continue

        new_sample_dir = os.path.join(VOICE_SAMPLES_DIR, str(new_id))
        os.makedirs(new_sample_dir, exist_ok=True)

        for script_id, file_path, created_at in old_samples:
            if not file_path or not os.path.exists(file_path):
                continue
            new_file_path = os.path.join(new_sample_dir, os.path.basename(file_path))
            if not os.path.exists(new_file_path):
                shutil.copy2(file_path, new_file_path)
            new_conn.execute(
                "INSERT INTO voice_samples (profile_id, script_id, file_path, created_at) VALUES (?,?,?,?)",
                (new_id, script_id, new_file_path, created_at)
            )
            migrated_samples += 1
    new_conn.commit()

    # ── Voice consent ──
    migrated_consent = 0
    try:
        consent_rows = old_conn.execute(
            "SELECT user_id, voice_consent_at FROM users WHERE voice_consent_at IS NOT NULL"
        ).fetchall()
    except sqlite3.OperationalError:
        consent_rows = []  # column doesn't exist on a very old chat.db

    for user_id, consented_at in consent_rows:
        external_user_id = str(user_id)
        new_conn.execute(
            "INSERT INTO voice_consent (client_id, external_user_id, consented_at) VALUES (?,?,?) "
            "ON CONFLICT(client_id, external_user_id) DO UPDATE SET consented_at=excluded.consented_at",
            (client_id, external_user_id, consented_at)
        )
        migrated_consent += 1
    new_conn.commit()

    old_conn.close()
    new_conn.close()

    print(f"[migrate] Client: {DEFAULT_CLIENT_NAME} (id={client_id})")
    print(f"[migrate] API key: {api_key}")
    print(f"[migrate]   -> put this in {src_dir}/voice_station_key.txt")
    print(f"[migrate] Migrated {migrated_profiles} cloned voice profile(s), "
          f"{migrated_samples} sample(s), {migrated_consent} consent record(s).")


if __name__ == "__main__":
    main()
