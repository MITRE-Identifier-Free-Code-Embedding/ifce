# identifier free code embedding

## Overview

This repository exists to reproduce the experiments performed in the paper Identifier Free Code Embedding Models for Scalable Search 

This experiment suite is designed to test and improve code embedding models at positioning their embeddings in space according to what a function does, and not how its variables are named or its identifiers.

This is accomplished by training with a contrastive learning objective that associates functions with their pseudo-c (decompiled) representations, instead of those with similar naming conventions or source representations.

Evaluation is accomplished through the following benchmarks:

* _Assemblage_ - source to decomplilation association (and vice versa)
* _Signsrch_ - algorithm constant association


## Requirements

This project uses the uv library. To get started, install uv via:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Using uv with any of the commands below, or calling:
```bash
uv sync
```

will automatically ensure your environment is configured correctly according to the pyproject.toml file.

Download the appropriate datasets and pass them into conduct_experiment.py. For instance, the Signsrch dataset:

```bash
wget https://aluigi.altervista.org/mytoolz/signsrch.sig.zip 
unzip signsrch.sig.zip
mv signsrch.sig ./data
```

Our variant of the Assemblage test dataset should be either included with the repo at /data/ or available from Huggingface (link TBD).

## Training

We perform distributed training with the [accelerate](https://huggingface.co/docs/accelerate/index) library. To add a default configuration, run:

```bash
uv run accelerate config
```

To train a new model with the train_model.py script:

```bash
uv run accelerate launch ifce/modeling/train_model.py --data_dir_path $TRAIN_ASSEMBLAGE_PATH
```

The resulting model should populate the ifce/models directory for its detection during evaluation.

## Evaluation

To run evaluation, use:

```bash
uv run python ifce/conduct_experiment.py --data_dir_path $TEST_ASSEMBLAGE_PATH --yaml_file_path $CONFIG_PATH --signsrch_path $SIGNSRCH_PATH
```

Evaluation results will populate in ifce/experiment_results. Some intermediate artifacts will be saved in the data directory. For example, latex tables will be generated that should look similar to:

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

## Contents

Scripts are present within the /ifce directory. Experimental results (figures, tables, etc.), and models are made available after running these scripts under /ifce/experiment_results and /ifce/models.

Although not strictly required by the scripts, data may be stored under the /data directory. Intermediate products, such as matrices of cosine similarity and stored embeddings, will also be stored in /data.

Configuration yamls are stored under /config. These correspond to scripts for timing, training, and testing experiments.

### Copyright

Copyright ©2026 The MITRE Corporation. ALL RIGHTS RESERVED. Public Release Case Number 26-0894.
