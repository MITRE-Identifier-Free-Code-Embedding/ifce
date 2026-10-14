#!/bin/bash
#SBATCH --job-name=ifce-eval
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#
# ============================================================================
# IFCE evaluation / experiment (single-process, single-GPU)
#
# Runs the Assemblage retrieval + Signsrch benchmarks over the models listed in
# config/experiment_config.yaml, via the `ifce-experiment` console script.
#
# BEFORE SUBMITTING, EDIT:
#   * DATA_DIR    -> path to your Assemblage TEST dataset (save_to_disk HF
#                    dataset with function_id/source_id/decompilation_id cols).
#                    Required.
#   * YAML_CONFIG -> experiment config listing the models to evaluate.
#   * SIGNSRCH    -> path to the signsrch .sig file (download per README Data).
#   * BATCH_SIZE  -> embedding batch size (tune to GPU memory).
#
# MODELS: every id in the YAML must be resolvable at eval time:
#   * HF-hub / vLLM models are pulled automatically;
#   * custom fine-tuned ids (e.g. BUSTA-IFCE-*Qwen3*) require a matching
#     checkpoint dir under outputs/checkpoints/ produced by train_slurm.sh.
#   The default config lists BOTH BUSTA-IFCE-Qwen3-0.6B AND
#   BUSTA-IFCE-Qwen3-0.6B-FP8 -- trim the list, or produce both checkpoints,
#   or eval will KeyError on the missing one.
#
# GPU: eval is single-process on ONE GPU. The Signsrch stage hardcodes
#      device="cuda", so a CUDA GPU is required (gpu:1). Large models in the
#      default config (Qwen3-Embedding-8B, SFR-Embedding-Mistral) may need a
#      high-memory GPU; TODO: request a larger GPU or trim the models list.
#
# NOTE: `logs/` must exist when the job STARTS. Create it before submitting:
#           mkdir -p logs && sbatch eval_slurm.sh
# NOTE: ifce-experiment also writes .pkl files (metrics_df.pkl, all_ranks.pkl,
#       group_aps_exp2.pkl, AP_dict_exps.pkl) into the CURRENT directory, so run
#       from the repo root. Figures/tables land in outputs/results/.
# ============================================================================

set -euo pipefail

# --- Config (EDIT ME) ---
YAML_CONFIG="config/experiment_config.yaml"
SIGNSRCH="data/signsrch.sig"
BATCH_SIZE=8

# TODO: set DATA_DIR to your save_to_disk TEST dataset. Defaults to the
#       $TEST_ASSEMBLAGE_PATH env var used in the README; override as needed.
DATA_DIR="${TEST_ASSEMBLAGE_PATH:-}"
if [[ -z "${DATA_DIR}" ]]; then
    echo "ERROR: DATA_DIR is empty. Set TEST_ASSEMBLAGE_PATH or edit DATA_DIR." >&2
    exit 1
fi

# --- Environment setup ---
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"
mkdir -p logs

# The default config/experiment_config.yaml includes bge-m3 (needs the `eval`
# extra / FlagEmbedding) and quantized/vLLM models (need the `serve` extra).
# TODO: drop --extra serve and/or --extra eval if you slim the models list.
uv sync --extra eval --extra serve

uv run --extra eval --extra serve ifce-experiment \
    --data_dir_path "${DATA_DIR}" \
    --yaml_file_path "${YAML_CONFIG}" \
    --signsrch_path "${SIGNSRCH}" \
    --batch_size "${BATCH_SIZE}"

# Results: outputs/results/ (figures, tables) and cosine matrices under data/.
