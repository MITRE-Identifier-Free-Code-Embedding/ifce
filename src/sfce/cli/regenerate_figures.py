# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The MITRE Corporation
"""Regenerate figures from saved pickles.

Useful for changing plot styling and generating new plots without rerunning the
full experiment pipeline. Reads the pickled intermediate results produced by a
prior run and re-renders the figures.
"""

import argparse
import pickle
from pathlib import Path

from sfce.config import yaml_safe_load
from sfce.eval.assemblage_retrieval import K_FOR_RECALLS, gen_products
from sfce.eval.signsrch_benchmark import gen_signsrch_products
from sfce.paths import (
    DEFAULT_EXPERIMENT_CONFIG,
    RESULTS_DIR,
    ensure_output_dirs,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results_dir",
        default=str(RESULTS_DIR),
        type=str,
        help="Directory holding the pickled intermediate results and where figures are written.",
    )
    parser.add_argument(
        "--config_path",
        default=str(DEFAULT_EXPERIMENT_CONFIG),
        type=str,
        help="Experiment YAML config (for the model list).",
    )
    args = parser.parse_args()

    ensure_output_dirs()
    results_dir = Path(args.results_dir)
    model_names = yaml_safe_load(args.config_path)["models"]

    with open(results_dir / "metrics_df.pkl", "rb") as file:
        metrics_df = pickle.load(file)
    with open(results_dir / "all_ranks.pkl", "rb") as file:
        all_ranks = pickle.load(file)

    gen_products(
        metrics_df=metrics_df,
        out_path=str(results_dir),
        all_ranks=all_ranks,
        k_for_recalls=K_FOR_RECALLS,
        model_names=model_names,
    )

    with open(results_dir / "AP_dict_exps.pkl", "rb") as file:
        AP_dict_exp1, AP_dict_exp2 = pickle.load(file)
    with open(results_dir / "group_aps_exp2.pkl", "rb") as file:
        group_aps_exp2 = pickle.load(file)

    gen_signsrch_products(AP_dict_exp1, AP_dict_exp2, group_aps_exp2, str(results_dir))


if __name__ == "__main__":
    main()
