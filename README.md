# semantic free code embedding

## Overview

What is Semantic Free Code Embedding?

Semantic Free Code Embedding is an experiment suite designed to test and improve code embedding models at positioning their embeddings in space according to what a function does, and not how its variables are named.

This is accomplished by training with a contrastive learning objective that associates functions with their pseudo-c representations, instead of those with similar naming conventions or source representations.

Evaluation is accomplished through the following benchmarks:

* Assemblage - source to decomplilation association (and vice versa)
* Signsrch - "magic" constant association


## Getting started

This project uses the uv library. To get started, install uv via. 
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Download the appropriate datasets into the dataset directory and pass them into conduct_experiment.py:

```
cd sfce
```

```bash
uv run python conduct_experiment.py --data_dir_path $ASSEMBLAGE_PATH --yaml_file_path $CONFIG_PATH --signsrch_path $SIGNSRCH_PATH
```

Results will populate in sfce/experiment_results.