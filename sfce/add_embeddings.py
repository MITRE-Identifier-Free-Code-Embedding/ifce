from models import MODEL_CLASS_MAP
from datasets import load_from_disk
from tqdm import tqdm
import yaml

"""
Note "id'd" data refers to data that has been id'd by 
1. function 
2. source function (starting from 0 to ds length) and 
3. decompilation function (starting from ds length to ds length * 2)
"""

def yaml_safe_load(filename:str):
    with open(filename, 'r') as file:
        data = yaml.safe_load(file)
        return data
    
def create_embeddings(data_directory: str, 
                      model_config_path: str, 
                      batch_size: int):
    models = []
    device = "cuda"
    yaml_config = yaml_safe_load(model_config_path)

    for model in yaml_config["models"]:
        if isinstance(model, str):
            models.append((model, MODEL_CLASS_MAP[model](device=device, name=model)))
        else:
            raise ValueError("Malformed Config")

    ds = load_from_disk(data_directory)
    batched_ds = ds.batch(batch_size = batch_size)

    for simple_name, local_model in models:

        # batch and process
        embedding_list = [[None] * len(ds) for _ in range(2)]
        for _, batch in enumerate(tqdm(batched_ds, desc="Embedding batches...")):
            for key_index, key in enumerate(['decompilation','source']):
                embeddings = local_model.embed_batch(batch[key])

                for index in range(len(batch[key])):
                    embedding_list[key_index][batch["function_id"][index]] = embeddings[index]
                

        for key_index, key in enumerate(['decompilation','source']):
            ds = ds.add_column(f"{simple_name}-embedding-{key}", embedding_list[key_index])
    
    new_ds_directory = f"{data_directory}_embedded"
    ds.save_to_disk(new_ds_directory)
    return new_ds_directory

if __name__ == "__main__":
    # Retained for test case
    DATA_DIR = "/home/ewolos/semantic-free-code-embedding/data/decompiled_summaries_TEST_copy_idd"
    CONFIG_PATH = "/home/ewolos/semantic-free-code-embedding/config/experiment_config.yaml"
    BATCH_SIZE = 64
    create_embeddings(data_directory = DATA_DIR, 
                      model_config_path = CONFIG_PATH, 
                      batch_size = BATCH_SIZE)
