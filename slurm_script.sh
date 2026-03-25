#!/bin/bash
#SBATCH --job-name=mer_kfold_v2
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --time=06:00:00
#SBATCH --account=education-eemcs-msc-dsait
#SBATCH --output=logs/kfold_dual_stream_%j.out
#SBATCH --error=logs/kfold_dual_stream_%j.err

# ── Load system modules ──────────────────────────────────────────────────────
module load 2025
module load python/3.11.9
module load cuda/12.1
# module load miniconda3

# ── Point to venv packages ───────────────────────────────────────────────────
export VIRTUAL_ENV=/scratch/smiyyapuram/medusa/.venv_medusa
export PATH=$VIRTUAL_ENV/bin:$PATH
export PYTHONPATH=$VIRTUAL_ENV/lib/python3.11/site-packages:$PYTHONPATH
# conda activate medusa
source $VIRTUAL_ENV/bin/activate

# ── Cache Directories (HuggingFace, PyTorch/timm) ────────────────────────────
export HF_HOME=/scratch/smiyyapuram/medusa/.cache/huggingface
export TORCH_HOME=/scratch/smiyyapuram/medusa/.cache/torch
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TIMM_FMT="safetensors"

# ── Sanity check ─────────────────────────────────────────────────────────────
echo "Hostname: $(hostname)"
echo "Starting at: $(date)"
python3 -c "import torch; print('CUDA available:', torch.cuda.is_available())"

# ── Move to working directory ────────────────────────────────────────────────
cd /scratch/smiyyapuram/medusa

# ── Run 10-fold CV with all data ─────────────────────────────────────────────
echo "Starting 10-fold CV (all data) at $(date)"
srun python trainKfold_single_stream.py \
    --kfold 10 \
    --all-data \
    --epochs 100


echo "Job finished at $(date)"