#!/bin/bash
#SBATCH --job-name=ifce-train
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

# --- Infrastructure Config ---
GPUS_PER_NODE=4
MODEL_NAME="Qwen/Qwen3-Embedding-0.6B"
MODEL_SHORT_NAME="Qwen"
BATCH_SIZE=64
GRAD_ACCUM=2
DATA_DIR="$HOME/semantic-free-code-embedding/data/decompiled_summaries_TRAIN"

# --- Environment Setup ---
# Use uv to run the script in a managed environment
# We use 'accelerate launch' to handle multi-GPU distribution
# Since SentenceTransformerTrainer is built on HF Trainer, 
# it integrates directly with accelerate.

mkdir -p logs

uv run accelerate launch \
    --multi_gpu \
    --num_processes $GPUS_PER_NODE \
    --mixed_precision bf16 \
    ifce/modeling/train_model.py \
    --model_name "$MODEL_NAME" \
    --model_short_name "$MODEL_SHORT_NAME" \
    --batch_size $BATCH_SIZE \
    --grad_accum_steps $GRAD_ACCUM \
    --data_dir_path "$DATA_DIR"
