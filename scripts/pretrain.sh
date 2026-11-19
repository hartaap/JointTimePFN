#!/bin/bash
# Pretraining, from scratch. ARM=joint is the joint model (joint NLL + lambda * marginal NLL),
# ARM=nll and ARM=beta are the marginal-only ablations with the per-cell Gaussian loss.
#   REPO=$PWD JOINT_LAMBDA=0.5 sbatch scripts/pretrain.sh                                 # reported model
#   REPO=$PWD JOINT_LAMBDA=0.5 JOINT_RANK=8 sbatch scripts/pretrain.sh                    # rank ablation
#   REPO=$PWD JOINT_LAMBDA=0.5 NO_ROPE=1 RUN_TAG=JointTimePFN_worope sbatch scripts/pretrain.sh
#   REPO=$PWD ARM=nll  sbatch scripts/pretrain.sh     # Marginal_JTPFN_nll  (log score)
#   REPO=$PWD ARM=beta sbatch scripts/pretrain.sh     # Marginal_JTPFN_beta (beta-NLL, 0.5)
# The marginal runs were submitted with --time=24:00:00 --mem=64G --cpus-per-task=10.

#SBATCH --job-name=pretrain
#SBATCH --output=logs/pretrain_%j.out
#SBATCH --error=logs/pretrain_%j.err
#SBATCH --time=30:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1

set -euo pipefail

REPO=${REPO:?set REPO to the repo root, e.g. REPO=$PWD sbatch scripts/...}
cd "$REPO"
mkdir -p logs

module purge
module load mamba
set +u
source activate ./env
set -u
export PYTHONUNBUFFERED=1

export PFN_MASK_PRESET=${PFN_MASK_PRESET:-foundation_mix}
export PFN_SEED=${PFN_SEED:-2021}

ARM=${ARM:-joint}
case "$ARM" in
  joint)
    MODEL="--model TimePFN_PPD_unified_joint --exp_name unified_joint"
    LAM=${JOINT_LAMBDA:?set JOINT_LAMBDA (0.5 for the reported model)}
    R=${JOINT_RANK:-32}
    ID="${RUN_TAG:-JointTimePFN}_l${LAM}_r${R}"
    LOSS="--joint_rank $R --joint_lambda $LAM"
    NUM_WORKERS=${NUM_WORKERS:-6}
    VAL_EVERY=${VAL_EVERY:-2000} ;;
  nll|beta)
    MODEL="--model TimePFN_PPD_unified_fold --exp_name unified_masked"
    BETA=$([ "$ARM" = nll ] && echo 0.0 || echo 0.5)
    ID="${RUN_TAG:-Marginal_JTPFN_${ARM}}"
    LOSS="--beta_nll $BETA"
    # the marginal runs used 10 workers and validated every 2500 steps; the interval is fixed at
    # 2500 in experiments/exp_unified_masked.py, so VAL_EVERY only records it
    NUM_WORKERS=${NUM_WORKERS:-10}
    VAL_EVERY=${VAL_EVERY:-2500} ;;
  *) echo "FATAL: ARM must be joint, nll or beta (got '$ARM')"; exit 1 ;;
esac
LR=${LR:-3e-4}

# architecture:
# --use_rope/--rope_base must match wherever a checkpoint is loaded (RoPE tables are not stored)
ARCH="$MODEL \
      --seq_len 96 --pred_len 96 --label_len 48 \
      --patch_size 32 --embed_dim 256 --d_model 1024 --d_ff 512 --e_layers 8 --n_heads 8 \
      --enc_in 20 \
      --use_rope --rope_base 10.0"

# NO_ROPE=1: the no-RoPE ablation, and note that its checkpoint must then be loaded without --use_rope
if [ "${NO_ROPE:-0}" = "1" ]; then
  ARCH="${ARCH/--use_rope --rope_base 10.0/}"
  echo "[abl] RoPE disabled"
fi

# data:
# TOTAL must equal the arrow's series count (the 80/10/10 split is taken from it)
ARROW=${ARROW:-"synthetic_60k_c20_na_02_joint.arrow"}
TOTAL=${TOTAL:-60000}
SYNTH_ROOT="./dataset/synthetics/"
[ -f "${SYNTH_ROOT}${ARROW}" ] || { echo "FATAL: ${SYNTH_ROOT}${ARROW} not found"; exit 1; }

# --stride 8: window stride, not the patch stride
DATA="--synthetic_data_path $ARROW --synthetic_root_path $SYNTH_ROOT \
      --total_synthetic_series $TOTAL --synth_split 801010 --synthetic_length 1024 \
      --stride 8 --min_channel 2 --val_series_cap 500"

# optimisation:
# the collate's RNG is unseeded, so pretraining is not reproducible bit level exactly.
BATCH=${BATCH:-16}
COMMON_TAIL="--joint_max_cells 0 --loss_norm sample --lradj synthetic \
             --val_every $VAL_EVERY --batch_size $BATCH --train_epochs 1 \
             --num_workers $NUM_WORKERS --patience 200 --des Exp --itr 1"

if [ -d "./checkpoints/$ID" ]; then
  echo "FATAL: ./checkpoints/$ID already exists. Move it aside or change RUN_TAG."
  exit 1
fi

echo "=== $ID  ARM=$ARM  $LOSS  lr=$LR  seed=$PFN_SEED  arrow=$ARROW  $(date) ==="
python -u -m run --is_training 1 $ARCH $DATA \
    --model_id "$ID" --checkpoints "./checkpoints/$ID" \
    $LOSS \
    --learning_rate $LR --pct_start 0.3 $COMMON_TAIL
echo "=== $ID done $(date) ==="
