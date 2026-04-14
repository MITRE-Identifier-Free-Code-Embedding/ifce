import matplotlib.pyplot as plt
import seaborn as sns
from analyze_sims import gen_products, K_FOR_RECALLS
from add_embeddings import yaml_safe_load
import pickle
import os
from pathlib import Path
from signsrch_to_dict import gen_signsrch_products, parse_signsrch

if __name__ == "__main__":

    CONFIG_PATH = f"{os.path.expanduser("~")}/semantic-free-code-embedding/config/experiment_config.yaml"
    model_names = yaml_safe_load(CONFIG_PATH)["models"]
    with open("metrics_df.pkl", 'rb') as file:
        metrics_df = pickle.load(file)

    with open("all_ranks.pkl", 'rb') as file:
        all_ranks = pickle.load(file)

    k_for_recalls = K_FOR_RECALLS

    out_path = Path(__file__).resolve().parent / Path("experiment_results")
    os.makedirs(out_path, exist_ok=True)

    #gen_products(metrics_df=metrics_df,
    #out_path=out_path,
    #all_ranks=all_ranks,
    #k_for_recalls=k_for_recalls,
    #model_names = model_names)

    with open("AP_dict_exps.pkl", 'rb') as file:
        AP_dict_exp1, AP_dict_exp2 = pickle.load(file)

    with open("group_aps_exp2.pkl", 'rb') as file:
        group_aps_exp2 = pickle.load(file)


    signsrch_path = "../data/signsrch.sig"
    sigs = parse_signsrch(signsrch_path)
    

    gen_signsrch_products(AP_dict_exp1, AP_dict_exp2, group_aps_exp2, out_path)