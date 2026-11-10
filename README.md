# Identifier-Free Code Embedding (IFCE)

Code accompanying the paper **"Identifier-Free Code Embedding Models for Scalable
Search."** It trains and evaluates code-embedding models that position their
embeddings according to *what a function does* rather than how its variables are
named or its identifiers are spelled.

This is accomplished by training with a contrastive learning objective that
associates functions with their pseudo-C (decompiled) representations, instead of
those with similar naming conventions or source representations.

Evaluation is performed through two retrieval benchmarks:

- **Assemblage** — source ↔ decompilation association
- **Signsrch** — algorithm-constant association

The importable package is named `ifce`.

## Installation

This project uses [uv](https://docs.astral.sh/uv/). Install it via:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

A baseline install is CPU/macOS friendly and is enough to import the package and
run the test suite:

```bash
uv sync
```

The heavy, GPU-oriented dependencies are split into optional extras so the core
install stays lightweight. Install the ones you need:

| Extra      | Provides                                              |
|------------|-------------------------------------------------------|
| `train`    | `accelerate`, `deepspeed`, `bitsandbytes`, `tensorboard` (distributed/quantized training) |
| `serve`    | `vllm` (high-throughput inference)                    |
| `eval`     | `flagembedding`, `openai` (extra evaluation backends) |
| `all`      | everything above                                      |

```bash
# e.g. for training + serving on a CUDA machine
uv sync --extra train --extra serve --extra eval
```

## Data

Download the Signsrch signature database into `data/`:

```bash
wget https://aluigi.altervista.org/mytoolz/signsrch.sig.zip
unzip signsrch.sig.zip
mv signsrch.sig ./data
```

Our train/test splits are derived from the
[Assemblage dataset](https://assemblage-dataset.net/) and provided as a Hugging
Face dataset saved with `save_to_disk`. The processed variant will be published
to the Hugging Face Hub (link forthcoming).

## Training

Distributed training uses [accelerate](https://huggingface.co/docs/accelerate/index).
Create a default configuration once:

```bash
uv run accelerate config
```

Then train (requires the `train` extra):

```bash
uv run accelerate launch -m ifce.modeling.train --data_dir_path $TRAIN_ASSEMBLAGE_PATH
```

Training is seeded (`--seed`, default 42) for reproducibility; note that GPU /
cuDNN / DeepSpeed kernels remain a source of irreducible nondeterminism. The
resulting checkpoint is written to `outputs/checkpoints/<--model_id>` (default
`BUSTA-IFCE-Qwen3-0.6B`), where it is auto-discovered during evaluation. The
`--model_id` must match the id used in `config/experiment_config.yaml`.

The default loss is cached cosine InfoNCE (`--loss cosine_infonce`) at
`--temperature 0.05`, preserving the baseline. To use the Lorentz hyperbolic
InfoNCE objective:

```bash
uv run accelerate launch -m ifce.modeling.train \
    --data_dir_path $TRAIN_ASSEMBLAGE_PATH \
    --model_id BUSTA-IFCE-Qwen3-0.6B-Hyperbolic \
    --loss hyperbolic_infonce --temperature 0.05 --curvature 0.1
```

This L2-normalizes pooled embeddings, applies the exponential map at the Lorentz
origin, and uses negative geodesic distance divided by temperature as the
InfoNCE logits. Curvature is fixed at `-c`, where `--curvature` supplies the
positive magnitude `c`. Geometry calculations run in FP32 even under mixed
precision. Temperature must be finite and positive. Curvature must be finite
and between `1e-30` and `100` for stable FP32 scoring.

Two flags control training speed. `--max_seq_length` (default 2048) caps the
context; leaving it at the base model's native length (32768 for
Qwen3-Embedding-0.6B) makes iterations slow and OOM-prone on long functions.
`--loss_mini_batch_size` (default 16) chunks the cached-loss embedding
computation; it affects speed and memory only, not the result, so raise it to go
faster or lower it if you hit OOM.

## Evaluation

### Evaluate a checkpoint (recommended)

Install the core dependencies and evaluate any local SentenceTransformer
checkpoint or Hugging Face model ID:

```bash
uv sync
uv run ifce eval /path/to/checkpoint --data-dir /path/to/test-dataset
```

If `TEST_ASSEMBLAGE_PATH` is already set, the command is simply:

```bash
uv run ifce eval /path/to/checkpoint --output results/new-loss.json
```

This runs the lightweight retrieval evaluation used for the rebuttal experiments.
It reports MRR, Recall@1, and Recall@10 for both Decompiled→Source and
Source→Decompiled, using a combined pool of all source and decompiled candidates
and excluding the query itself. The matching target is the opposite representation
in the same dataset row. Ties use the original rebuttal script's optimistic rank:
one plus the number of strictly higher scores. The full paper pipeline below sorts
scores and can order ties differently; do not assume the two implementations are
bit-for-bit identical.

The dataset must be saved with `datasets.save_to_disk` and contain `source` and
`decompilation` string columns. A saved Dataset works directly. For a DatasetDict,
the `test` split is selected if present, otherwise the only split; use `--split`
to select explicitly. The complete selected split is used, without sampling or
deduplication. With the original test set this is 29,500 pairs / 59,000 candidates.

Defaults: maximum input length **2048 tokens**, encoding batch size **128**, and
ranking chunks of **512 queries**. CUDA is selected when available, otherwise CPU.
Keep the dataset and token limit identical when comparing loss functions. Lower
`--batch-size` if embedding runs out of GPU memory, or `--rank-batch-size` if ranking
runs out of memory. Example with explicit settings:

```bash
CUDA_VISIBLE_DEVICES=0 uv run ifce eval /path/to/checkpoint \
    --max-seq-length 2048 --batch-size 32 --output results/new-loss.json
```

Cosine scoring is the default. For a checkpoint trained with hyperbolic loss,
select Lorentz scoring and match the curvature magnitude used in training:

```bash
uv run ifce eval /path/to/checkpoint \
    --distance-type hyperbolic --curvature 0.1 --output results/hyperbolic.json
```

Hyperbolic scoring uses the training objective's normalized Lorentz mapping and
FP32 negative geodesic distance. Both scoring modes rank larger scores first and
use the same combined candidate pool, self-match exclusion, and tie policy.

JSON output records both directions' metrics, dataset path/split, checkpoint,
token limit, batch sizes, scoring mode, curvature (for hyperbolic scoring), device,
pool size, tie policy, and elapsed time. Model registration, vLLM, Signsrch files,
and the `eval`/`serve` extras are not required.
Only enable `--trust-remote-code` for checkpoints whose code you trust.
Use `uv run ifce eval --help` for all options; `uv run python -m ifce eval` is
equivalent. Existing `ifce-train`, `ifce-experiment`, and `ifce-time` commands remain
available.

### Advanced: full paper benchmark suite

Use `ifce-experiment` when you need the full Assemblage and Signsrch benchmarks,
model comparisons, figures, and LaTeX tables. This requires a model YAML
configuration and the Signsrch signature database described above:

```bash
uv run ifce-experiment \
    --data_dir_path $TEST_ASSEMBLAGE_PATH \
    --yaml_file_path config/experiment_config.yaml \
    --signsrch_path data/signsrch.sig
```

Both Assemblage and Signsrch support cosine (default) and hyperbolic retrieval.
Add `--distance_type hyperbolic --curvature 0.1` to `ifce-experiment` for
Lorentz scoring, matching the curvature magnitude used in training. Alternatively,
configure defaults in the experiment YAML:

```yaml
evaluation:
  distance_type: hyperbolic
  curvature: 0.1
```

CLI options override YAML. Hyperbolic scoring uses the same normalized Lorentz
mapping and FP32 negative geodesic distance as training. Both metrics rank larger
scores first.

Figures, LaTeX tables, and intermediate pickles are saved in `outputs/results/cosine/`
for cosine evaluation and in `outputs/results/hyperbolic_c0.1/` for hyperbolic
evaluation, with the subdirectory reflecting the selected curvature. Cached score
matrices follow the same layout under `outputs/distances/`: for example,
`outputs/distances/cosine/<model>.parquet` or
`outputs/distances/hyperbolic_c0.1/<model>.parquet`. Embeddings remain in the
`<dataset_path>_embedded` dataset and can be reused across metrics. New runs use
these directories; older caches in `data/` are not reused automatically.

To re-render figures from saved pickles without rerunning evaluation, run
`uv run python -m ifce.cli.regenerate_figures`, which defaults to
`outputs/results/cosine/`. For hyperbolic results, add
`--results_dir outputs/results/hyperbolic_c0.1`, using the directory for your
selected curvature.

A generated results table looks similar to:

**Table: MRR, MRR@k and Recall@k for Searching Decompiled to Source, Decompilation and Source Candidates.**
| Model                                 | MRR    | Recall@1 | Recall@5 | Recall@10 | MRR@10 |
|---------------------------------------|--------|----------|----------|-----------|--------|
| BinSeek-Embedding                     | 0.0802 | 0.0594   | 0.1005   | 0.1221    | 0.0771 |
| **BUSTA-IFCE-Qwen3-0.6B**             | **0.4104** | **0.3065** | **0.5176** | **0.6427** | **0.3983** |
| BUSTA-IFCE-Qwen3-0.6B-FP8           | 0.4083 | 0.3047   | 0.5162   | 0.6406    | 0.3961 |
| Qwen3-Embedding-0.6B                  | 0.0569 | 0.0434   | 0.0714   | 0.0822    | 0.0551 |
| bge-small-en-v1.5                     | 0.0075 | 0.0056   | 0.0095   | 0.0113    | 0.0072 |
| bge-m3                                | 0.0049 | 0.0034   | 0.0062   | 0.0078    | 0.0046 |
| Qwen3-Embedding-8B                    | 0.0258 | 0.0167   | 0.0346   | 0.0433    | 0.0242 |
| embeddinggemma-300m                   | 0.0196 | 0.0127   | 0.0259   | 0.0327    | 0.0184 |
| snowflake-arctic-embed-l-v2.0         | 0.0096 | 0.0061   | 0.0126   | 0.0162    | 0.0089 |
| SFR-Embedding-Mistral                 | 0.0671 | 0.0508   | 0.0842   | 0.0980    | 0.0649 |
| jina-embeddings-v2-base-code          | 0.0001 | 0.0000   | 0.0000   | 0.0000    | 0.0000 |
| nomic-embed-code                      | 0.0209 | 0.0166   | 0.0253   | 0.0290    | 0.0203 |

## Repository layout

```
src/ifce/          importable package
  cli/             runnable entry points (ifce eval, ifce-experiment, ifce-train, ifce-time, regenerate_figures)
  modeling/        embedding model wrappers + training
  processing/      embedding generation and distance-matrix computation
  eval/            Assemblage retrieval, Signsrch benchmark, OOD analyses
  visualization/   plotting
  paths.py         canonical project directories (override via IFCE_* env vars)
config/            experiment / timing / training YAML configs
data/              inputs (signsrch.sig); generated parquet/embeddings (gitignored)
outputs/           generated artifacts: checkpoints/, results/, distances/ (gitignored)
tests/             import smoke + unit tests
```

## Development

See [CONTRIBUTING.md](CONTRIBUTING.md). In short: `uv sync`, then
`uv run ruff check . --fix`, `uv run ruff format .`, and `uv run pytest`.

## Citation

If you use this software or its results, please cite the paper (see
[CITATION.cff](CITATION.cff)):

```bibtex
@misc{ifce2026,
  title  = {Identifier-Free Code Embedding Models for Scalable Search},
  author = {Wolos, Eric and Doyle, Michael},
  year   = {2026}
}
```

## License

Licensed under the Apache License, Version 2.0 — see [LICENSE](LICENSE) and
[NOTICE](NOTICE).

Copyright © 2026 The MITRE Corporation. Public Release Case Number 26-0894.
