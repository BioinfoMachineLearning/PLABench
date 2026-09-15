#!/bin/bash
# Chain-rule ablation, step 4+5: wait out the four stage-1 encoders, then run the
# multi-view integration and score the arm. Runs unattended so the three cards stay busy.
# no `set -u`: conda's own activation hooks reference unset variables and would abort here
ROOT=/bmlfast/Lyuwei/0.Projects/MixingDTA
W=/bmlfast/Lyuwei/0.Projects/PLABench/scratch/concat_ablation
L=$W/logs
DS=PDBbind_Refined_91_concat
RR=$ROOT/MEETA/results_refined_91_concat

source ~/miniforge3/etc/profile.d/conda.sh
conda activate MixingDTA
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd $ROOT/MEETA || exit 1

echo "[$(date +%F_%T)] waiting for the four stage-1 encoders"
# Both conditions matter. "No training process" alone is not enough: case 6 is chained after
# case 1 on GPU 0, so there is a moment with no trainer running and only three checkpoints
# written. Requiring all four checkpoints as well closes that window. The stall counter is
# the escape hatch for the opposite case -- a case that dies without ever writing one.
stall=0
while true; do
  have=0
  for sub in results_none results_all_pair results_drug results_reversed; do
    [ -f "$RR/$sub/$DS/1_fold_valid_best_checkpoint.pth" ] && have=$((have + 1))
  done
  n=$(pgrep -fc "run_refined_91_concat_training.py")
  [ "$n" -eq 0 ] && [ "$have" -eq 4 ] && break
  if [ "$n" -eq 0 ]; then
    stall=$((stall + 1))
    if [ $stall -ge 10 ]; then
      echo "[$(date +%F_%T)] aborting: no trainer running but only $have/4 checkpoints"
      for sub in results_none results_all_pair results_drug results_reversed; do
        [ -f "$RR/$sub/$DS/1_fold_valid_best_checkpoint.pth" ] || echo "  missing: $sub"
      done
      exit 1
    fi
  else
    stall=0
  fi
  sleep 120
done

echo "[$(date +%F_%T)] all four encoders done; starting integration"
python run_refined_91_concat_integration.py --gpu 0 > $L/tr_concat_integration.log 2>&1
if [ ! -f $RR/result_integration/$DS/1_fold_valid_best_checkpoint.pth ]; then
  echo "[$(date +%F_%T)] integration produced no checkpoint; see tr_concat_integration.log"
  exit 1
fi

echo "[$(date +%F_%T)] scoring the concat arm on the four external test sets"
cd $ROOT || exit 1
python $W/evaluate_arm.py --arm concat --gpu 0 > $L/eval_concat.log 2>&1
echo "[$(date +%F_%T)] done"
cat $L/eval_concat.log
