#!/bin/bash
# capped_run.sh <command...>  -- test/import wrapper of the kit, memory-capped (4 GiB), no GPU.
#
# Order 910 (coordinator rule 03.10. ~22:45Z, /spinning/gpu-arb/deskq/RULES.md): the ONLY pytest path on this rig is
# /spinning/gpu-arb/pytest_gedeckelt.sh -- it keeps the lock rules itself and yields to a pending boot (BOOT_PENDING).
# Never a bare flock on pytest.lock. So:
#   * a pytest command (`<python> ... -m pytest <args>`) goes through pytest_gedeckelt.sh (python and PYTHONPATH of the
#     caller are handed over; -q / -p no:cacheprovider are dropped, the wrapper sets them itself);
#   * every other command (import probe, launcher dry-run) is not a pytest: it only runs under the 4 GiB cap, without the lock.
export CUDA_VISIBLE_DEVICES=""
args=("$@"); n=${#args[@]}; idx=-1
for ((i = 0; i + 1 < n; i++)); do
  if [ "${args[i]}" = "-m" ] && [ "${args[i+1]}" = "pytest" ]; then idx=$i; break; fi
done
if [ $idx -ge 0 ] && [ $idx -le 3 ]; then
  rest=(); skip=0
  for ((j = idx + 2; j < n; j++)); do
    a=${args[j]}
    if [ $skip = 1 ]; then skip=0; [ "$a" = "no:cacheprovider" ] && continue; rest+=("-p" "$a"); continue; fi
    case $a in -q) continue;; -p) skip=1; continue;; esac
    rest+=("$a")
  done
  export PYTEST_PYTHON="${args[0]}"
  export PYTEST_PYTHONPATH="${PYTHONPATH:-python:test/registered/unit/weg2}"
  exec /spinning/gpu-arb/pytest_gedeckelt.sh "${rest[@]}"
fi
exec systemd-run --scope -q -p MemoryMax=4G -- "$@"
