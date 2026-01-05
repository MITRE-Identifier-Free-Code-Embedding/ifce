from models import MODEL_CLASS_MAP
from time import time
from datasets import load_from_disk
from datasets.utils import disable_progress_bars, enable_progress_bars
from tqdm import tqdm
import yaml

DATA_DIR = "/home/ewolos/semantic-free-code-embedding/data/decompiled_summaries_TEST_copy_idd"
CONFIG_DIR = "/home/ewolos/semantic-free-code-embedding/config/experiment_config.yaml"
FIRST_N = 100
BATCH_SIZE = 16

def yaml_safe_load(filename:str):
    with open(filename, 'r') as file:
        data = yaml.safe_load(file)
        return data
    
def main():
    models = []
    device = "cuda"
    yaml_config = yaml_safe_load()
    # breakpoint()
    for model in yaml_config["models"]:
        #if isinstance(model, dict):
        #models.append(model, MODEL_CLASS_MAP[model](**yaml_config["models"][model], device=device, name=model))
        if isinstance(model, str):
            models.append((model, MODEL_CLASS_MAP[model](device=device, name=model)))
        else:
            raise ValueError("Malformed Config")

    
    ds = load_from_disk(DATA_DIR).select(range(FIRST_N))

    for simple_name, local_model in models:

        # batch and process
        batched_ds = ds.batch(batch_size = BATCH_SIZE)
        embedding_list = [[None] * len(ds)] * 2
        for _, batch in enumerate(tqdm(batched_ds, desc="Embedding batches...")):
            for key_index, key in enumerate(['decompilation','source']):
                embeddings = local_model.embed_batch(batch[key])

                for index in range(len(batch[key])):
                    embedding_list[key_index][batch["function_id"][index]] = embeddings[index]
                

        for key_index, key in enumerate(['decompilation','source']):
            ds = ds.add_column(f"{simple_name}-embedding-{key}", embedding_list[key_index])
    

    ds.save_to_disk(f"{DATA_DIR}_embedded")

if __name__ == "__main__":
    main()
