import argparse
from processing.add_embeddings import create_embeddings
from processing.find_distances import find_distances
from analyze_sims import analyze_sims
from signsrch_exps import run_signsrch_exps
import os


def arg_parse():
    parser = argparse.ArgumentParser(
        description="Conduct Assemblage and Signsrch Experiments with Embedding Models.",
        epilog="",
    )

    parser.add_argument(
        "--data_dir_path",
        default=f"{os.path.expanduser('~')}/semantic-free-code-embedding/data/decompiled_summaries_TEST_copy_idd",
        type=str,
        help="The path to the directory to process.",
    )

    parser.add_argument(
        "--yaml_file_path",
        default=f"{os.path.expanduser('~')}/semantic-free-code-embedding/config/experiment_config.yaml",
        type=str,
        help="The path to the yaml config.",
    )

    parser.add_argument(
        "--signsrch_path",
        default=f"{os.path.expanduser('~')}/semantic-free-code-embedding/data/signsrch.sig",
        type=str,
        help="The path to the Signsrch Sig file.",
    )

    # Optional argument/flags
    parser.add_argument(
        "--batch_size", default=8, type=int, help="Determine embedding batch size."
    )

    # Currently deprecated
    parser.add_argument(
        "--ood_test", default=False, type=bool, help="Determine embedding batch size."
    )

    args = parser.parse_args()
    return args


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
        save_euclidean=args.ood_test,
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
