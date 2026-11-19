#!/bin/bash
#SBATCH --job-name=gen_rank_arrow
#SBATCH --output=logs/gen_rank_arrow_%j.out
#SBATCH --error=logs/gen_rank_arrow_%j.err
#SBATCH --time=09:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=18

# The pretraining arrow: 60k series, 20 channels, 1024 steps (CPU only).
#   REPO=$PWD sbatch scripts/generate_synthetic.sh

REPO=${REPO:?set REPO to the repo root}
cd "$REPO"
mkdir -p logs
mkdir -p dataset/synthetics

module purge
module load mamba
source activate ./env

python -m synthetic_data_generation.LMC_Synth_uusi \
      -N 60000 \
      -C 20 \
      -L 1024 \
      -J 5 \
      -M 0.1 \
      -X 10.0 \
      -P 0.2 \
      -NA \
      -O "synthetic_60k_c20_na_02_joint.arrow" \
      -D "./dataset/synthetics/"

echo "done: $(date)"
