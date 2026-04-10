#!/bin/bash
#SBATCH --job-name=graphguard_q1
#SBATCH --partition=gpu-short
#SBATCH --qos=gpu-short
#SBATCH --gres=gpu:a100-80gb:1
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=custom_question_%j.out
#SBATCH --error=custom_question_%j.err

set -eo pipefail

echo "=== GraphGuard Custom Question Run ==="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Host: $(hostname)"
echo "Start: $(date)"

cd /nfs_home/users/vintia/GraphGuard

# Avoid inherited prompt hooks in batch shells.
unset PROMPT_COMMAND || true

source /etc/profile.d/modules.sh
module purge
module load miniconda/default
module load cuda/12.8

source /nfs_home/software/miniconda/etc/profile.d/conda.sh
set +u
conda activate graphguard
set -u

export PYTHONPATH="/nfs_home/users/vintia/GraphGuard/tokenizer:${PYTHONPATH:-}"
cd tokenizer

# Override these at submit time with: --export=ALL,QUESTION="...",GOLD="...",SAMPLE_ID="q_001",CONTEXT=""
QUESTION="${QUESTION:-What is the capital of Australia?}"
CONTEXT="${CONTEXT:-}"
GOLD="${GOLD:-Canberra}"
SAMPLE_ID="${SAMPLE_ID:-q_001}"
DATASET_NAME="${DATASET_NAME:-CustomUserPrompt}"

python collectors/collect_benchmark.py \
  --custom-prompt "${QUESTION}" \
  --custom-context "${CONTEXT}" \
  --custom-gold-answer "${GOLD}" \
  --custom-sample-id "${SAMPLE_ID}" \
  --custom-dataset-name "${DATASET_NAME}"

echo "End: $(date)"
