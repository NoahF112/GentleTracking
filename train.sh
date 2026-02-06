#!/usr/bin/env bash
set -euo pipefail

# ===== Global Configuration =====
# PROJECT="axell-wppr/gentle_humanoid"
# PROJECT="axell-wppr/motion_tracking"
PROJECT="ligo125438-tum/gentle_humanoid"
NPROC=1
SCRIPT="scripts/train.py"
curr_date=$(date +"%Y%m%d")
RUN_NAME=${1:-$curr_date}

run_pipeline() {
  local TASK="$1" TAG="$2" SUFFIX="$3"

  local ID_TRAIN="${TAG}_train_${SUFFIX}"
  local ID_ADAPT="${TAG}_adapt_${SUFFIX}"
  local ID_FINETUNE="${TAG}_finetune_${SUFFIX}"

  # ---------- TRAIN ----------
  cmd=(torchrun --nproc_per_node="$NPROC" "$SCRIPT"
    task="$TASK" +exp=train
    wandb.id="$ID_TRAIN"
  )
  echo ">>> ${cmd[*]}"; "${cmd[@]}"

  # ---------- ADAPT ----------
  cmd=(torchrun --nproc_per_node="$NPROC" "$SCRIPT"
    task="$TASK" +exp=adapt
    checkpoint_path="run:${PROJECT}/${ID_TRAIN}"
    wandb.id="$ID_ADAPT"
  )
  echo ">>> ${cmd[*]}"; "${cmd[@]}"

  # ---------- FINETUNE ----------
  cmd=(torchrun --nproc_per_node="$NPROC" "$SCRIPT"
    task="$TASK" +exp=finetune
    checkpoint_path="run:${PROJECT}/${ID_ADAPT}"
    wandb.id="$ID_FINETUNE"
  )
  echo ">>> ${cmd[*]}"; "${cmd[@]}"
}

# run_pipeline "G1/G1_tracking" "track" "0112.3"
run_pipeline "G1/G1_tracking" "track" "${RUN_NAME}"

# run_pipeline "G1/G1_gentle" "gentle" "1215"
# run_pipeline "G1/G1_no_force" "noforce" "1215"
# run_pipeline "G1/G1_extreme_force" "extremeforce" "1215"
