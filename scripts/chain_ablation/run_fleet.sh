#!/bin/bash
# Chain-rule ablation for LLF and DeepDTA: 2 arms x 3 seeds x 2 models = 12 training runs.
#
# Both arms use window 4700, which is what MEETA already used and is wide enough that
# neither arm truncates anything (longest chain maxes at 1287, concatenation at 4638). The
# published checkpoints -- LLF at window 1200, DeepDTA at 2000 -- are a third reference arm
# and are not retrained here.
#
# Deliberately NOT `set -u`: conda's activation hooks reference unbound variables.
set -eo pipefail

W=/bmlfast/Lyuwei/0.Projects/PLABench/scratch/concat_ablation
LLF_PY=/bmlfast/Lyuwei/0.Projects/LLF/.conda/bin/python
DD_ENV=/home/lwfvx/miniforge3/envs/deepdta
# the deepdta env ships libstdc++.so.6.0.34 but no libstdc++.so.6 symlink, so the loader
# falls back to /lib64 and pandas fails on GLIBCXX_3.4.29; preload it instead of touching
# the user's env
export LD_PRELOAD=$DD_ENV/lib/libstdc++.so.6.0.34

queue_llf() {   # $1 = gpu, rest = "arm:seed" pairs
  local gpu=$1; shift
  for job in "$@"; do
    local arm=${job%%:*} seed=${job##*:}
    local tag="llf_${arm}_L4700_s${seed}"
    echo "[gpu$gpu] start $tag  $(date +%H:%M:%S)"
    cd "$W" && $LLF_PY llf_train.py --arm "$arm" --max_seq_len 4700 \
        --seed "$seed" --gpu "$gpu" > "$W/logs/$tag.log" 2>&1
    echo "[gpu$gpu] done  $tag  $(date +%H:%M:%S)"
  done
}

queue_dd() {
  local gpu=$1; shift
  for job in "$@"; do
    local arm=${job%%:*} seed=${job##*:}
    local tag="dd_${arm}_L4700_s${seed}"
    echo "[gpu$gpu] start $tag  $(date +%H:%M:%S)"
    cd "$W" && $DD_ENV/bin/python deepdta_train.py --arm "$arm" --seqlen 4700 \
        --seed "$seed" --gpu "$gpu" > "$W/logs/$tag.log" 2>&1
    echo "[gpu$gpu] done  $tag  $(date +%H:%M:%S)"
  done
}

case "$1" in
  q0) queue_llf 0 concat:0 concat:1 ;;
  q1) queue_llf 1 longest:0 longest:1 ;;
  q2) queue_llf 2 concat:2 longest:2 ;;
  qd) queue_dd 0 concat:1 concat:2 longest:0 longest:1 longest:2 ;;
  *)  echo "usage: $0 {q0|q1|q2|qd}" >&2; exit 2 ;;
esac
