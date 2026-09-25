#!/bin/sh
# Docker-HEALTHCHECK fuer beide Modi -- ENTWURF (27B-Sitz R, 24.09.2026).
# MODE=weg2: Front /health (200 nur, wenn beide Gruppen /health 200 liefern und state != STOP,
#            front.py:2756-2767). Readiness (state == serving) prueft der Testtreiber ueber /weg2/state.
# sonst:     /health des August-Servers auf ${PORT:-30000}.
if [ "${MODE:-server}" = "weg2" ]; then
  exec curl -fsS -m 8 -o /dev/null http://127.0.0.1:30030/health
fi
exec curl -fsS -m 8 -o /dev/null "http://127.0.0.1:${PORT:-30000}/health"
