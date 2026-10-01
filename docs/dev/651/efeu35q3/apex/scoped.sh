#!/bin/bash
# efeu-TP14 (operator rule after the 21:16 OOM): every build or test on the
# laptop runs capped -- a 4 GiB memory scope, lowest CPU and I/O priority.
#   apex/scoped.sh python setup.py build_ext --inplace
exec systemd-run --scope -q -p MemoryMax=4G nice -n 19 ionice -c3 "$@"
