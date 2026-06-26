# Identifier-Free Code Embedding (IFCE)

Code accompanying the paper **"Identifier Free Code Embedding Models for Scalable
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

Our variant of the Assemblage train/test dataset is a Hugging Face dataset saved
with `save_to_disk`. <!-- TODO: publish the final Hugging Face URL and dataset license. -->

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
resulting checkpoint is written under `outputs/checkpoints/`, where it is
auto-discovered during evaluation.

## Evaluation

```bash
uv run ifce-experiment \
    --data_dir_path $TEST_ASSEMBLAGE_PATH \
    --yaml_file_path config/experiment_config.yaml \
    --signsrch_path data/signsrch.sig
```

Evaluation results (figures, LaTeX tables) populate `outputs/results/`;
intermediate cosine-similarity matrices and embeddings are stored under
`data/` and `outputs/distances/`. Figures can be re-rendered from saved pickles
without rerunning the experiment via `uv run python -m ifce.cli.regenerate_figures`.

A generated results table looks similar to:

**Table: MRR, MRR@k and Recall@k for Searching Decompiled to Source, Decompilation and Source Candidates.**
| Model                                 | MRR    | Recall@1 | Recall@5 | Recall@10 | MRR@10 |
|---------------------------------------|--------|----------|----------|-----------|--------|
| BinSeek-Embedding                     | 0.0802 | 0.0594   | 0.1005   | 0.1221    | 0.0771 |
| **mitRE-embed-Qwen-0.6B**             | **0.4104** | **0.3065** | **0.5176** | **0.6427** | **0.3983** |
| mitRE-embed-quant-Qwen-0.6B           | 0.4083 | 0.3047   | 0.5162   | 0.6406    | 0.3961 |
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
  cli/             runnable entry points (ifce-experiment, ifce-train, ifce-time, regenerate_figures)
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
@inproceedings{ifce2026,
  title     = {Identifier Free Code Embedding Models for Scalable Search},
  author    = {The MITRE Corporation},
  year      = {2026}
  % TODO: authors, venue (NeurIPS 2026), and DOI / arXiv id
}
```

## License

Licensed under the Apache License, Version 2.0 — see [LICENSE](LICENSE) and
[NOTICE](NOTICE).

Copyright © 2026 The MITRE Corporation. Public Release Case Number 26-0894.
