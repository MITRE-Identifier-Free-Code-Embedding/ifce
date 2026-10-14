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
#
# ============================================================================
# IFCE multi-GPU training (distributed via HuggingFace accelerate)
#
# BEFORE SUBMITTING, EDIT:
#   * DATA_DIR       -> path to your Assemblage TRAIN dataset (save_to_disk
#                       HF dataset containing a "train" split). Required.
#   * MODEL_ID       -> trained-model name; also the checkpoint dir name under
#                       outputs/checkpoints/<MODEL_ID>. MUST match the id you
#                       list in config/experiment_config.yaml so eval can find it.
#   * GPUS_PER_NODE  -> keep in sync with #SBATCH --gres=gpu:N above.
#   * SBATCH resources (--cpus-per-task/--mem/--time) -> tune to your job.
#
# NOTE: `logs/` must exist when the job STARTS or SLURM cannot write %j.out/%j.err.
#       Create it before submitting:   mkdir -p logs && sbatch train_slurm.sh
# ============================================================================

set -euo pipefail

# --- Infrastructure config (EDIT ME) ---
GPUS_PER_NODE=4                                  # keep in sync with --gres=gpu:N
MODEL_NAME="Qwen/Qwen3-Embedding-0.6B"           # base model to fine-tune
MODEL_ID="BUSTA-IFCE-Qwen3-0.6B"                 # checkpoint name; must match config
BATCH_SIZE=64
GRAD_ACCUM=2
SEED=42

# TODO: set DATA_DIR to your save_to_disk TRAIN dataset. Defaults to the
#       $TRAIN_ASSEMBLAGE_PATH env var used in the README; override as needed.
DATA_DIR="${TRAIN_ASSEMBLAGE_PATH:-}"
if [[ -z "${DATA_DIR}" ]]; then
    echo "ERROR: DATA_DIR is empty. Set TRAIN_ASSEMBLAGE_PATH or edit DATA_DIR." >&2
    exit 1
fi

# --- Environment setup ---
# Run from the repo root (the dir containing pyproject.toml / this script).
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"
mkdir -p logs

# accelerate/deepspeed/bitsandbytes live in the optional `train` extra now,
# so sync it before launching (idempotent).
uv sync --extra train

# accelerate launch spawns one process per GPU. SentenceTransformerTrainer is a
# HF Trainer subclass, so it integrates directly with accelerate. The trainer is
# launched as an installed MODULE (-m ifce.modeling.train), not a file path.
uv run --extra train accelerate launch \
    --multi_gpu \
    --num_processes "${GPUS_PER_NODE}" \
    --mixed_precision bf16 \
    -m ifce.modeling.train \
    --data_dir_path "${DATA_DIR}" \
    --model_name "${MODEL_NAME}" \
    --model_id "${MODEL_ID}" \
    --batch_size "${BATCH_SIZE}" \
    --grad_accum_steps "${GRAD_ACCUM}" \
    --seed "${SEED}"

# Resulting checkpoint: outputs/checkpoints/${MODEL_ID}
