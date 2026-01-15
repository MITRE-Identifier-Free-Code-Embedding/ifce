import argparse
from add_embeddings import create_embeddings
from find_cosine_sim import find_cosine_sim
from analyze_sims import analyze_sims

def arg_parse():
    parser = argparse.ArgumentParser(
        description="Conduct MRR and Recall Experiments with Embedding Models.",
        epilog=""
    )

    parser.add_argument(
        "data_dir_path",
        default="/home/ewolos/semantic-free-code-embedding/data/decompiled_summaries_TEST_copy_idd",
        type=str,
        help="The path to the directory to process."
    )

    parser.add_argument(
        "yaml_file_path",
        default="/home/ewolos/semantic-free-code-embedding/config/experiment_config.yaml",
        type=str,
        help="The path to the yaml config."
    )

    # Optional argument/flag
    parser.add_argument(
        "--batch_size",
        default=64,
        type=int,
        help="Determine embedding batch size."
    )

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = arg_parse()
    
    processed_embeddings_ds = create_embeddings(data_directory=args.data_dir_path, 
                      model_config_path=args.yaml_file_path,
                       batch_size=args.batch_size)

    print(f"Processed Embeddings at {processed_embeddings_ds}")
    model_names, sim_files = find_cosine_sim(data_input_dir=processed_embeddings_ds, model_config_path=args.yaml_file_path)
    print(f"Processed cosine distances at {dict(zip(model_names, sim_files))}")
    
    #model_names = ["bge-m3", "bge-small-en-v1.5"]
    #sim_files = ["../data/cosine_bge-m3.parquet", "../data/cosine_bge-small-en-v1.5.parquet"] 
    analyze_sims(model_names=model_names, cosine_parquet_paths=sim_files)
    print("experiment complete!")


    