#!/bin/bash
#SBATCH --job-name=prep400
#SBATCH --output=logs/prep400_%x_%j.out
#SBATCH --error=logs/prep400_%x_%j.err
#SBATCH --time=12:00:00
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8

# Build the MIMIC-IV benchmark cells; CPU unless a GPU is requested (sbatch --gpus=1 ...).
set -euo pipefail
#   REPO=$PWD sbatch scripts/prep_mimic.sh

REPO=${REPO:?set REPO to the repo root, e.g. REPO=$PWD sbatch scripts/...}
cd "$REPO"

mkdir -p logs
module purge
module load mamba
set +u
source activate ./env
set -u

: "${MIMIC_NPZ:?set MIMIC_NPZ=<dataset>/mimic_iv/test/mimic.npz}"
[ -f "$MIMIC_NPZ" ] || { echo "FATAL: $MIMIC_NPZ not found"; exit 1; }
MAXW=${MAXW:-409}
OUTDIR=${OUTDIR:-bench_io_ehr3h_400w}
CONDS=${CONDS:-A,F,C,DRH}
MAXT=${MAXT:-10}
# The arms the two tables name, each with its own checkpoint in benchmark.models.MODELS.
MODELS=${MODELS:-JointTimePFN_l0.5_r32,JointTimePFN_worope_l0.5_r32,JointTimePFN_l0.5_r4,JointTimePFN_l0.5_r8,JointTimePFN_l0.5_r16,JointTimePFN_l0.5_r64,JointTimePFN_l0.5_r128,JointTimePFN_l0.0_r32,JointTimePFN_l0.25_r32,JointTimePFN_l0.75_r32,JointTimePFN_l1.0_r32,Marginal_JTPFN_nll,Marginal_JTPFN_beta,timepfn_original}

ARCH="--model TimePFN_PPD_unified_fold --data ETTh1 --exp_name unified_masked --features M \
      --seq_len 96 --pred_len 96 --label_len 48 --enc_in 20 \
      --e_layers 8 --d_model 1024 --d_ff 512 --n_heads 8 --patch_size 32 --embed_dim 256 \
      --use_rope --rope_base 10.0"

_DEV=$(python -c "import torch;print('cuda' if torch.cuda.is_available() else 'cpu')")
echo "=== prep mimic_iv  windows=$MAXW  targets=$MAXT  conds=$CONDS  device=$_DEV  out=$OUTDIR  $(date) ==="
python -u -m benchmark.prep $ARCH \
    --models "$MODELS" --mimic_npz "$MIMIC_NPZ" \
    --max_windows "$MAXW" --max_targets "$MAXT" \
    --conditions "$CONDS" \
    --out_dir "$OUTDIR" \
    --rate "${RATE:-0.3}"
echo "=== prep done $(date) ==="
