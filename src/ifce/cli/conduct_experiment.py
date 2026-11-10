# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The MITRE Corporation
import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

import yaml

from ifce.config import yaml_safe_load
from ifce.eval.assemblage_retrieval import analyze_sims
from ifce.eval.signsrch_benchmark import run_signsrch_exps
from ifce.modeling.models.registry import register_checkpoint
from ifce.paths import DEFAULT_EXPERIMENT_CONFIG, DEFAULT_SIGNSRCH_PATH
from ifce.processing.add_embeddings import create_embeddings
from ifce.processing.find_distances import (
    evaluation_results_dir,
    evaluation_settings,
    find_distances,
)


def arg_parse():
    parser = argparse.ArgumentParser(
        description="Conduct Assemblage and Signsrch experiments with embedding models.",
    )
    parser.add_argument(
        "checkpoint_path",
        nargs="?",
        help="Local Qwen3-0.6B checkpoint directory; evaluates only this model.",
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
        "--distance_type",
        choices=("cosine", "hyperbolic"),
        default=None,
        help="Retrieval metric; overrides evaluation.distance_type in YAML (default: cosine).",
    )
    parser.add_argument(
        "--curvature",
        type=float,
        default=None,
        help="Positive Lorentz curvature magnitude; overrides YAML (default: 0.1).",
    )
    return parser.parse_args()


def conduct_experiment():

    args = arg_parse()
    if args.checkpoint_path:
        model_name = register_checkpoint(args.checkpoint_path)
        config = yaml_safe_load(args.yaml_file_path)
        config["models"] = [model_name]
        with TemporaryDirectory(prefix="ifce-eval-") as directory:
            config_path = Path(directory) / "experiment.yaml"
            config_path.write_text(yaml.safe_dump(config))
            args.yaml_file_path = str(config_path)
            _run_experiment(args)
    else:
        _run_experiment(args)


def _run_experiment(args):
    distance_type, curvature = evaluation_settings(
        yaml_safe_load(args.yaml_file_path), args.distance_type, args.curvature
    )
    out_path = evaluation_results_dir(distance_type, curvature)
    out_path.mkdir(parents=True, exist_ok=True)

    processed_embeddings_ds = create_embeddings(
        data_directory=args.data_dir_path,
        model_config_path=args.yaml_file_path,
        batch_size=args.batch_size,
        out_path=str(out_path),
    )

    model_names, sim_files = find_distances(
        data_input_dir=processed_embeddings_ds,
        model_config_path=args.yaml_file_path,
        distance_type=distance_type,
        curvature=curvature,
    )

    analyze_sims(
        model_names=model_names,
        cosine_parquet_paths=sim_files,
        data_input_dir=processed_embeddings_ds,
        out_path=out_path,
    )
    run_signsrch_exps(
        model_config_path=args.yaml_file_path,
        signsrch_path=args.signsrch_path,
        distance_type=distance_type,
        curvature=curvature,
    )

    print("Completed search metric analysis.")
    print("Experiment complete!")


if __name__ == "__main__":
    conduct_experiment()
