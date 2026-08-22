#!/usr/bin/env bash
# Serve the glasses web app over HTTPS so phone browsers allow live camera access.
# Usage: scripts/serve_https.sh            (PORT=8443 by default)
# Then open https://<printed-ip>:PORT on the phone and accept the self-signed cert warning once.
set -euo pipefail
cd "$(dirname "$0")/.."
PORT="${PORT:-8443}"
UVICORN="${UVICORN:-$([ -x .venv/bin/uvicorn ] && echo .venv/bin/uvicorn || echo uvicorn)}"

mkdir -p certs
if [ ! -f certs/dev-key.pem ] || [ ! -f certs/dev-cert.pem ]; then
  openssl req -x509 -newkey rsa:2048 -nodes -days 3650 -subj "/CN=glasses-dev" \
    -addext "subjectAltName=DNS:localhost,IP:127.0.0.1" \
    -keyout certs/dev-key.pem -out certs/dev-cert.pem 2>/dev/null
  echo "generated certs/dev-cert.pem (self-signed)"
fi

IP="$(ipconfig getifaddr en0 2>/dev/null || ipconfig getifaddr en1 2>/dev/null || hostname -I 2>/dev/null | awk '{print $1}' || true)"
echo "Phone URL: https://${IP:-<this-machine-ip>}:${PORT}   (accept the certificate warning once)"
exec "$UVICORN" glasses_detector.api:app --host 0.0.0.0 --port "$PORT" \
  --ssl-keyfile certs/dev-key.pem --ssl-certfile certs/dev-cert.pem
