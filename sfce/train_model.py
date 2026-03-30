from datasets import load_from_disk
from copy import deepcopy
from sentence_transformers.losses import CachedMultipleNegativesRankingLoss, MultipleNegativesRankingLoss
from sentence_transformers.training_args import BatchSamplers, MultiDatasetBatchSamplers
from sentence_transformers import SentenceTransformerTrainingArguments, SentenceTransformerTrainer, SentenceTransformer
from transformers import AutoTokenizer, DataCollatorWithPadding, DataCollatorForSeq2Seq


import os
from transformers import TrainerCallback

def train(ds_path : str, model_name: str):

    # model init
    model = SentenceTransformer(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Data init
    ds = load_from_disk(ds_path)["train"]
    ds = ds.rename_columns({"source":"anchor",
                            "decompilation":"positive"}).remove_columns("summary") # may use this later
    
    half_length = int(round(model.max_seq_length / 2)) # keeping the context limited for this example model
    
    # Used to truncate inputs into desired lengths.
    encode_decode = lambda x: tokenizer.decode(tokenizer.encode(x, 
                                                                truncation=True, 
                                                                max_length=half_length), 
                                                                skip_special_tokens=True)
    ds = ds.map(lambda x: {"anchor": encode_decode(x["anchor"]),
                            "positive": encode_decode(x["positive"])
                           }) # I'm not totally convinced the data is getting truncated, so we do it manually.

    ds = ds.select_columns(["anchor", "positive"])
    split_ds = ds.train_test_split(test_size=0.01)
    train_ds, val_ds = split_ds["train"], split_ds["test"]

    mini_batch_size = 1 # In nearly every case, this should always be 1
    batch_size = 64 # This is treated a little bit like gradient accumulation, a little bit differently
    eval_batch_size = 4
    grad_accum_steps = 2 # TODO maybe pull this from the config automatically?

    # misc init
    loss = CachedMultipleNegativesRankingLoss(model, mini_batch_size=mini_batch_size)
    # TODO eventually implement MatryoshkaLoss

    run_name = f"mitRE_embed_{model_name}_source_anchor"

    args = SentenceTransformerTrainingArguments(
        dataloader_drop_last=True,
        torch_empty_cache_steps=1, # prevents OOM
        output_dir=f"models/mitRE_embed_Qwen/{run_name}",
        # Optional training parameters:
        num_train_epochs=1,
        gradient_accumulation_steps=grad_accum_steps,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=eval_batch_size,
        learning_rate=2e-5,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        fp16=False,  
        bf16=True,  
        batch_sampler=BatchSamplers.NO_DUPLICATES,  # MultipleNegativesRankingLoss benefits from no duplicate samples in a batch
        # Optional tracking/debugging parameters:
        eval_strategy="steps",
        eval_steps=10,
        save_strategy="steps",
        save_steps=10,
        save_total_limit=2,
        logging_steps=1,
        logging_first_step=True,
        run_name=run_name,  # Used if `wandb`, `tensorboard`, or `neptune`, etc. is installed
        report_to="tensorboard",
        skip_memory_metrics=False,
        save_safetensors=True
    )

    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        loss=loss,
        # data_collator=data_collator,
        # evaluator=evaluator, TODO resolve at some point
    )
    trainer.train()


if __name__ == "__main__":
    DS_PATH = "/home/ewolos/semantic-free-code-embedding/data/decompiled_summaries_TRAIN"
    model_name = "Qwen/Qwen3-Embedding-0.6B"
    train(ds_path=DS_PATH, model_name = model_name)