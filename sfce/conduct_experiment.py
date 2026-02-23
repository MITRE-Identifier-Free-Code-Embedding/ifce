import argparse
from add_embeddings import create_embeddings
from find_distances import find_distances
from analyze_sims import analyze_sims
from get_stats_for_f1 import get_stats_for_f1

def arg_parse():
    parser = argparse.ArgumentParser(
        description="Conduct MRR and Recall Experiments with Embedding Models.",
        epilog=""
    )

    parser.add_argument(
        "--data_dir_path",
        default="/home/ewolos/semantic-free-code-embedding/data/decompiled_summaries_TEST_copy_idd",
        type=str,
        help="The path to the directory to process."
    )

    parser.add_argument(
        "--yaml_file_path",
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
    
    #processed_embeddings_ds = create_embeddings(data_directory=args.data_dir_path, 
    #                  model_config_path=args.yaml_file_path,
    #                   batch_size=args.batch_size)

    #print(f"Processed Embeddings at {processed_embeddings_ds}")
    # model_names, sim_files
    processed_embeddings_ds = "../data/decompiled_summaries_TEST_copy_idd_embedded/"
    #model_names, sim_files = find_distances(data_input_dir=processed_embeddings_ds, model_config_path=args.yaml_file_path)
    #print(f"Processed cosine distances at {dict(zip(model_names, sim_files))}")
    model_names = ["mitRE_embed_Qwen_1",
    "Qwen3-Embedding-0.6B",
    "mitRE_embed_Qwen_2",
    "bge-small-en-v1.5"]
    sim_files = [f"../data/cosine_{model_name}.parquet" for model_name in model_names]
    model_names_doubled = sorted(model_names*2)
    sim_files_doubled = []
    for i, model_name in enumerate(model_names_doubled):
        if i % 2 == 1:
            continue
        for dist_type in ["cosine", "euclidean"]:
            sim_files_doubled.append(f"../data/{dist_type}_{model_name}.parquet")
            
    #get_stats_for_f1(model_list=model_names_doubled,
    #                 similarity_files=sim_files_doubled)
    print("Calibrated OOD detection")
    
    analyze_sims(model_names=model_names, cosine_parquet_paths=sim_files, data_input_dir= processed_embeddings_ds)
    print("Completed search metric analysis")
    print("Experiment complete!")


    