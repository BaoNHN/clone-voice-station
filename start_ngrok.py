# start_ngrok.py
# Exposes this app (port 8090) publicly for a demo -- a temporary stand-in until
# real deployment (see thesis future-work notes). NOT for production use: no auth
# in front of ngrok's tunnel beyond whatever this app itself enforces (manager
# login / client API key / STT Lab guest login).
#
# Set your own token via:
#   Windows:      set NGROK_AUTHTOKEN=your_token
#   macOS/Linux:  export NGROK_AUTHTOKEN=your_token
# Get one from https://dashboard.ngrok.com/tunnels/authtokens -- never hardcode
# it in this file (see rag-legal-assistant/start_ngrok.py's own history for why:
# a hardcoded token there leaked into git history and had to be rotated).
import os
import sys
import time

import ngrok

AUTHTOKEN = os.getenv("NGROK_AUTHTOKEN")
if not AUTHTOKEN:
    print("NGROK_AUTHTOKEN environment variable is not set.", file=sys.stderr)
    print("Get a token from https://dashboard.ngrok.com/tunnels/authtokens and set it first:", file=sys.stderr)
    print("  Windows:      set NGROK_AUTHTOKEN=your_token", file=sys.stderr)
    print("  macOS/Linux:  export NGROK_AUTHTOKEN=your_token", file=sys.stderr)
    sys.exit(1)

listener = ngrok.forward(8090, authtoken=AUTHTOKEN)
print(f"\n✅ Public URL: {listener.url()}\n")
print("Paste this into any app's VOICE_STATION_URL env var to point it at this")
print("tunnel instead of http://127.0.0.1:8090 (requires restarting that app).")
print("Press Ctrl+C to stop.")

try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("Stopped.")
