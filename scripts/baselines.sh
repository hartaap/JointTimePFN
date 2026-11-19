#!/bin/bash
# The external baselines on the shared prep cells. DEVICE defaults to cuda, so
# request the GPU on the submit line, or set DEVICE=cpu:
#   TabImpute, TabPFN-TS (GPU):   sbatch --gpus=1 scripts/baselines.sh tabimpute
#   Chronos-2, TiRex-2 (CPU):     sbatch --export=ALL,DEVICE=cpu -c 16 scripts/baselines.sh chronos

#SBATCH --job-name=nat_full
#SBATCH --output=logs/nat_full_%x_%j.out
#SBATCH --error=logs/nat_full_%x_%j.err
#SBATCH --time=20:00:00
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8

set -euo pipefail
ARM=${1:?usage: sbatch scripts/baselines.sh tabimpute|chronos|tabpfnts|tirex}
REPO=${REPO:?set REPO to the repo root, e.g. REPO=$PWD sbatch scripts/...}
cd "$REPO"
mkdir -p logs

IN_DIR=${IN_DIR:-bench_io_ehr3h_400w}
CONDS=${CONDS:-A,F,C,DRH}
DEVICE=${DEVICE:-cuda}
# TabPFN-TS sampling frequency: the MIMIC-IV bin width
FREQ=${FREQ:-3h}
# Chronos-2: amazon/chronos-2 (120M, tag chronos2) or autogluon/chronos-2-small (28M, tag chronos2small)
MODEL=${MODEL:-amazon/chronos-2}
# the output prefix benchmark.merge expects: tabpfnts_, tabimpute_, chronos2_, chronos2small_, tirex2_
TAG=${TAG:-}

module purge
module load mamba

case "$ARM" in
  tabimpute)
    ( set +u; source activate ./env_tabimpute; set -u
      python -u -m baselines.tabimpute --in_dir "$IN_DIR" --conditions "$CONDS" \
          --device "$DEVICE" --tag "${TAG:-tabimpute}" )
    ;;
  tabpfnts)
    ( set +u; source activate ./env_tabpfn; set -u
      export TABPFN_DISABLE_TELEMETRY=1
      python -u -m baselines.tabpfn_ts --in_dir "$IN_DIR" --conditions "$CONDS" \
          --freq "$FREQ" --tag "${TAG:-tabpfnts}" )
    ;;
  chronos)
    # env_chronos is a venv, not a conda env
    [ -x env_chronos/bin/python ] || { echo "FATAL: env_chronos/bin/python missing"; exit 1; }
    HF_HOME=${HF_HOME:-$REPO/.hf_cache} env_chronos/bin/python -u -m baselines.chronos2 \
        --in_dir "$IN_DIR" --conditions "$CONDS" --device "$DEVICE" \
        --model "$MODEL" --tag "${TAG:-chronos2}"
    ;;
  tirex)
    # env_tirex is a venv (Python 3.12)
    [ -x env_tirex/bin/python ] || { echo "FATAL: env_tirex/bin/python missing"; exit 1; }
    env_tirex/bin/python -u -m baselines.tirex --in_dir "$IN_DIR" --conditions "$CONDS" \
        --device "$DEVICE" --tag "${TAG:-tirex2}"
    ;;
  *) echo "unknown arm: $ARM"; exit 1 ;;
esac

echo "=== $ARM full done $(date) ==="
