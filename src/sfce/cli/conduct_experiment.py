# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The MITRE Corporation
import argparse

from sfce.eval.assemblage_retrieval import analyze_sims
from sfce.eval.signsrch_benchmark import run_signsrch_exps
from sfce.paths import DEFAULT_EXPERIMENT_CONFIG, DEFAULT_SIGNSRCH_PATH
from sfce.processing.add_embeddings import create_embeddings
from sfce.processing.find_distances import find_distances


def arg_parse():
    parser = argparse.ArgumentParser(
        description="Conduct Assemblage and Signsrch experiments with embedding models.",
    )

    parser.add_argument(
        "--data_dir_path",
        required=True,
        type=str,
        help="Path to the Assemblage test dataset (a Hugging Face dataset saved with save_to_disk).",
    )

    parser.add_argument(
        "--yaml_file_path",
        default=str(DEFAULT_EXPERIMENT_CONFIG),
        type=str,
        help="Path to the experiment YAML config (defaults to config/experiment_config.yaml).",
    )

    parser.add_argument(
        "--signsrch_path",
        default=str(DEFAULT_SIGNSRCH_PATH),
        type=str,
        help="Path to the Signsrch .sig file (defaults to data/signsrch.sig).",
    )

    # Optional argument/flags
    parser.add_argument(
        "--batch_size", default=8, type=int, help="Embedding batch size."
    )

    parser.add_argument(
        "--save_euclidean",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Also compute and save Euclidean distance matrices in addition to cosine.",
    )

    return parser.parse_args()


def conduct_experiment():

    args = arg_parse()

    processed_embeddings_ds = create_embeddings(
        data_directory=args.data_dir_path,
        model_config_path=args.yaml_file_path,
        batch_size=args.batch_size,
    )

    model_names, sim_files = find_distances(
        data_input_dir=processed_embeddings_ds,
        model_config_path=args.yaml_file_path,
        save_euclidean=args.save_euclidean,
    )

    analyze_sims(
        model_names=model_names,
        cosine_parquet_paths=sim_files,
        data_input_dir=processed_embeddings_ds,
    )
    run_signsrch_exps(
        model_config_path=args.yaml_file_path, signsrch_path=args.signsrch_path
    )

    print("Completed search metric analysis.")
    print("Experiment complete!")


if __name__ == "__main__":
    conduct_experiment()
