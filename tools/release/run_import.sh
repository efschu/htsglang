#!/bin/bash
F=${RELEASE_WORK:-/spinning/flliper/work/release}
mkdir -p $F; K=${RELEASE_KIT:-$(cd "$(dirname "$0")" && pwd)}; cd $K && HOME=$F/home PYTHONPATH=${IMPORT_TREE:-$F/wt-probe}/python exec ${RIG_TEST_WRAP:-$K/capped_run.sh} ${PY:-/spinning/htsglang-gpu/.venv/bin/python} -W ignore import_check.py
