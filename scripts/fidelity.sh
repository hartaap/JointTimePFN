#!/bin/bash
# The fidelity grid: one array task per line of tasks/fid_tasks.txt (480 lines); the file must not
# change while the array runs. The datasets are fidelity_data.zip in the Release.
#   REPO=$PWD sbatch --array=0-479%20 scripts/fidelity.sh

#SBATCH --job-name=fidelity
#SBATCH --output=logs/fidelity_%A_%a.out
#SBATCH --error=logs/fidelity_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1

set -euo pipefail
REPO=${REPO:?set REPO to the repo root, e.g. REPO=$PWD sbatch scripts/...}
cd "$REPO"
mkdir -p logs

# one task per line, '#' comments ignored: <arm marg|joint> <band imd|oomd> <N> <seed> <cond A|F|C|DRH>
TASKS=${TASKS:-tasks/fid_tasks.txt}
[ -f "$TASKS" ] || { echo "FATAL: $TASKS missing -- see the task-file format above"; exit 1; }
LINE=$(grep -v '^#' "$TASKS" | sed -n "$((${SLURM_ARRAY_TASK_ID:-0} + 1))p")
[ -n "$LINE" ] || { echo "no task at index ${SLURM_ARRAY_TASK_ID:-0}"; exit 0; }
read -r ARM BAND N SEED COND <<< "$LINE"
echo "task ${SLURM_ARRAY_TASK_ID:-0}: arm=$ARM band=$BAND N=$N seed=$SEED cond=$COND  $(date)"

module purge
module load mamba
set +u
source activate ./env
set -u

# the oracle and NUTS Cholesky are BLAS-bound. Only the PFN forward uses the GPU
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export PYTHONUNBUFFERED=1
# fidelity/nuts.py keeps NUTS on the CPU unless this is set
export PFN_NUTS_DEVICE=${PFN_NUTS_DEVICE:-cuda}
# validated at import; the fidelity masks come from harness.build_grids
export PFN_MASK_PRESET=foundation_mix

# checkpoint and architecture (both arms score the joint checkpoint, the marginal arm its implied marginals):
# --use_rope/--rope_base must match the checkpoint: RoPE tables are not in the state_dict, so a mismatch loads cleanly and silently computes something else
CKPT=${CKPT:-'checkpoints/JointTimePFN_l0.5_r32/*/checkpoint_joint_nll.pth'}
shopt -s nullglob
_ck=( $CKPT )
if [ ${#_ck[@]} -ne 1 ]; then
  echo "FATAL: CKPT glob matched ${#_ck[@]} files, need exactly 1:"; printf '   %s\n' "${_ck[@]}"
  exit 1
fi
echo "checkpoint: ${_ck[0]}"
R=${JOINT_RANK:-32}
ARCH="--model TimePFN_PPD_unified_joint --data ETTh1 --exp_name unified_joint --features M \
      --seq_len 96 --pred_len 96 --label_len 48 --enc_in 20 \
      --e_layers 8 --d_model 1024 --d_ff 512 --n_heads 8 --patch_size 32 --embed_dim 256 \
      --use_rope --rope_base 10.0"

# NUTS: 4 chains x (1000 warm-up + 1000 draws) on 5 windows, one target. NUTS_WIN is the bootstrap
# cluster count. The marginal arm runs NUTS only at N <= NUTS_MAX_N (cost cubic per step)
NUTS_WARM=${NUTS_WARM:-1000} NUTS_SAMP=${NUTS_SAMP:-1000} NUTS_WIN=${NUTS_WIN:-5} NUTS_MAX_N=${NUTS_MAX_N:-6}
NU="--nuts --nuts_warmup $NUTS_WARM --nuts_samples $NUTS_SAMP --nuts_chains ${NUTS_CHAINS:-4} \
    --nuts_max_windows $NUTS_WIN --nuts_max_targets 1"
echo "nuts: chains=${NUTS_CHAINS:-4} warmup=$NUTS_WARM samples=$NUTS_SAMP windows=$NUTS_WIN max_n=$NUTS_MAX_N"

D="dataset/toy_lmc_composed_k5_${BAND}_N${N}_d${SEED}"
[ -d "$D" ] || { echo "FATAL: $D missing -- generate the composed dataset first"; exit 1; }

if [ "$ARM" = marg ]; then
  MARG_DIR=${MARG_DIR:-marginal_c4_1000}
  mkdir -p "$MARG_DIR"
  [ "$N" -le "$NUTS_MAX_N" ] || NU=""
  python -u -m fidelity.marginal $ARCH --joint_rank $R --ckpts "$CKPT" \
      --composed_glob "$D" \
      --data_seeds 0 --max_windows 15 --max_targets 5 --conditions "$COND" \
      $NU \
      --out "${MARG_DIR}/composed_${BAND}_N${N}_d${SEED}_${COND}.json"
else
  JOINT_DIR=${JOINT_DIR:-joint_c4_1000}
  mkdir -p "$JOINT_DIR"
  _nt=5; [ "$N" -lt "$_nt" ] && _nt=$N
  python -u -m fidelity.joint $ARCH --ppd_ckpt UNUSED \
      --joint_rank "$R" --unified_ckpt "${_ck[0]}" \
      --csv_path "$D/toy_oracle.csv" --oracle_params "$D/params.npz" \
      --condition "$COND" --max_windows 15 --max_cells "${MAX_CELLS:-60}" \
      --targets "$(seq -s, 0 $((_nt - 1)))" $NU \
      --out "${JOINT_DIR}/composed_${BAND}_N${N}_d${SEED}_${COND}_l05.json"
fi
echo "=== task done $(date) ==="
