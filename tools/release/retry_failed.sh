#!/bin/bash
# retry_failed.sh <worktree> <nodeid> [tries=3]  -- Auftrag 910 / Punkt 2: ein im Kit neu roter Test (Zeittests unter Last,
# z. B. test_a_real_prewarm_holds_the_loop_under_120_ms) wird EINZELN, gedeckelt (RIG_TEST_WRAP, Standard capped_run.sh:
# flock pytest.lock + systemd-run MemoryMax=4G, kein GPU) und seriell bis zu <tries> mal wiederholt.
# Exit 0 = mindestens ein gruener Lauf (Lastflake; die Zeile "RETRY-PASS <n>/<tries>" nennt den Versuch), Exit 1 = alle
# Laeufe rot (echter Befund, das Kit bricht ab). Das HOME-Verzeichnis ist das des Testlaufs (run_tests.sh, $RELEASE_WORK/home-<wt>).
W=$1; N=$2; TRIES=${3:-3}
F=${RELEASE_KIT:-$(cd "$(dirname "$0")" && pwd)}; WORK=${RELEASE_WORK:-/spinning/flliper/work/release}
H=$WORK/home-$(basename "$W"); [ -d "$H" ] || { echo "RETRY-NOHOME $H"; exit 1; }
cd "$W" || exit 1
for i in $(seq 1 "$TRIES"); do
  out=$(HOME=$H PYTHONPATH=$W/python ${RIG_TEST_WRAP:-$F/capped_run.sh} ${PY:-/spinning/htsglang-gpu/.venv/bin/python} -m pytest -q -p no:cacheprovider --color=no "$N" 2>&1 | tail -3)
  if echo "$out" | grep -qE '[0-9]+ passed' && ! echo "$out" | grep -qE '[0-9]+ (failed|error)'; then
    echo "RETRY-PASS $i/$TRIES $N"; exit 0
  fi
  echo "RETRY-RED $i/$TRIES $N: $(echo "$out" | tail -1)"
done
exit 1
