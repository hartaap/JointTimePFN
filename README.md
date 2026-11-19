# JointTimePFN

Code for *JointTimePFN: amortized joint prediction on partially observed multivariate time series* (NeurIPS 2026 Workshop on Foundation Models for Temporal Systems).

## Install

    pip install -r requirements.txt

The baselines (Chronos-2, TiRex-2, TabPFN-TS, TabImpute) each need their own environment.

## Checkpoints and fidelity data

Attached to the GitHub Release. Unzip `checkpoints.zip` and `fidelity_data.zip` in the repo root.

## Data

MIMIC-IV is not included. It requires PhysioNet credentialed access; the extraction query is `benchmark/mimic_query.sql`.

## Running

Run modules from the repo root (`python -m <package>.<module>`). Slurm scripts in `scripts/` need `REPO=$PWD`. Conditions A–D in the paper are `A`, `F`, `C` and `DRH` in the code.

Pretraining:

    REPO=$PWD sbatch scripts/generate_synthetic.sh
    REPO=$PWD JOINT_LAMBDA=0.5 sbatch scripts/pretrain.sh

Fidelity (Tables 1, 10 and 14):

    sbatch --export=ALL,REPO=$PWD --array=0-479%20 scripts/fidelity.sh
    python -m tables.fidelity_panels --ar_dir joint_c4_1000 --drop_failed_ref

MIMIC-IV (Table 2), after saving the query result as CSV:

    python -m benchmark.mimic_prepare --src mimic_iv_data/bq-results-*.csv --out dataset/mimic_iv
    MAXW=409 OUTDIR=bench_io MIMIC_NPZ=dataset/mimic_iv/test/mimic.npz REPO=$PWD sbatch scripts/prep_mimic.sh
    IN_DIR=bench_io REPO=$PWD sbatch --gpus=1 scripts/baselines.sh <tabimpute|tabpfnts|chronos|tirex>
    python -m benchmark.merge --in_dir bench_io --drop_wins bad_windows.txt --out bench.json
    python -m tables.real --json bench.json --metric crps_d --metric2 mse

## Credit

Builds on [TimePFN](https://github.com/egetaga/TimePFN) (Taga et al., 2025). See `LICENSE`.
