"""
engine/server_log.py
Central logging so the manager dashboard can show "what happened in the system"
(GET /manager/logs, see app.py + templates/dashboard.html) instead of requiring
someone to be watching the raw terminal -- which is what prompted this: repeated
back-and-forth pasting terminal output just to diagnose a stuck training run.

Deliberately scoped to this app's own events (training progress, Colab/local
fallback decisions, errors) via a single named logger, not the process's full
stdout/stderr or uvicorn's own request-access logs -- those are noisy (every
poll request) and not what a manager actually wants from a page titled "what
happened". get_logger() replaces the print() calls that used to scatter this
same information across engine/voice_engine.py, voice/rvc_client.py,
voice/rvc_local.py and voice/stt.py, console-only and gone once the terminal
scrolled past.
"""

import logging
import logging.handlers
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_FILE = os.path.join(BASE_DIR, "server.log")

_LOGGER_NAME = "clone-voice-station"
_configured = False


def _configure_once():
    global _configured
    if _configured:
        return
    _configured = True

    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    file_handler = logging.handlers.RotatingFileHandler(
        LOG_FILE, maxBytes=5_000_000, backupCount=3, encoding="utf-8"
    )
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    # A dedicated named logger, not the root logger -- keeps this independent of
    # uvicorn's own logging config (which reconfigures the root/uvicorn.* loggers
    # when uvicorn.run() starts) so there's no import-order dependency to get
    # wrong, and propagate=False keeps our lines from also going through any
    # handlers uvicorn attaches to root.
    logger = logging.getLogger(_LOGGER_NAME)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)


def get_logger() -> logging.Logger:
    """Shared logger for the whole app -- callers keep their own "[Voice]"/
    "[RVC]"/"[RVC-local]"/"[STT]"-style prefix in the message text itself
    (unchanged from the print() calls this replaces) rather than using
    Python's logger-name hierarchy, so migrating a print(f"[X] ...") call
    site is a mechanical one-line change."""
    _configure_once()
    return logging.getLogger(_LOGGER_NAME)


def read_recent_lines(max_lines: int = 500) -> list[str]:
    """Tail of the current log file, oldest first (same order they were
    written) -- used by GET /manager/logs. Only reads the active file, not
    rotated .1/.2/.3 backups; max_lines=500 comfortably fits a while before
    that matters for a page meant for "what's happening now"."""
    if not os.path.exists(LOG_FILE):
        return []
    with open(LOG_FILE, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()
    return [line.rstrip("\n") for line in lines[-max_lines:]]
