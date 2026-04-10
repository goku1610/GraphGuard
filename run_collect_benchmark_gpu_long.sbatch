#!/bin/bash
#SBATCH --job-name=collect_benchmark_gpu_long
#SBATCH --partition=gpu-long
#SBATCH --qos=gpu-long
#SBATCH --gres=gpu:a100-80gb:1
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=collect_benchmark_%j.out
#SBATCH --error=collect_benchmark_%j.err

set -eo pipefail

echo "=== GraphGuard collect_benchmark (gpu-long) ==="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Host: $(hostname)"
echo "Start: $(date)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

cd /nfs_home/users/vintia/GraphGuard

module purge
module load miniconda/default
module load cuda/12.8
source /nfs_home/software/miniconda/etc/profile.d/conda.sh
# Conda activate/deactivate hooks may reference unset vars; avoid nounset here.
set +u
conda activate graphguard
set -u
export PYTHONPATH="/nfs_home/users/vintia/GraphGuard/tokenizer:${PYTHONPATH:-}"

python -V
python -c "import torch; print('torch', torch.__version__, 'cuda_available', torch.cuda.is_available())"

cd tokenizer
LIMIT="${LIMIT:-5}"
DATASET="${DATASET:-math}"
SPLIT="${SPLIT:-test}"

python collectors/collect_benchmark.py --dataset "${DATASET}" --split "${SPLIT}" --limit "${LIMIT}"

echo "End: $(date)"
