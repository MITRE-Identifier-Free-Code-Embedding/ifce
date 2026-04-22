from datasets import load_from_disk
from sentence_transformers.losses import (
    CachedMultipleNegativesRankingLoss,
)
from sentence_transformers.training_args import BatchSamplers
from sentence_transformers import (
    SentenceTransformerTrainingArguments,
    SentenceTransformerTrainer,
    SentenceTransformer,
)
from transformers import AutoTokenizer


import os
from pathlib import Path
import argparse


def arg_parse():
    parser = argparse.ArgumentParser(
        description="Conduct Assemblage Training.",
        epilog="",
    )

    parser.add_argument(
        "--data_dir_path",
        default=f"{os.path.expanduser('~')}/semantic-free-code-embedding/data/decompiled_summaries_TRAIN",
        type=str,
        help="The path to the directory to process.",
    )

    parser.add_argument(
        "--model_name",
        default="Qwen/Qwen3-Embedding-0.6B",
        type=str,
    )

    parser.add_argument(
        "--model_short_name",
        default="Qwen",
        type=str,
    )

    parser.add_argument(
        "--grad_accum_steps",
        default=2,
        type=int,
        help="Gradient accumulatioin steps. Should match the train config",
    )

    parser.add_argument("--batch_size", default=64, type=int, help="Batch size")

    args = parser.parse_args()
    return args


def train(
    ds_path: str,
    model_name: str,
    model_short_name: str,
    grad_accum_steps: int,
    batch_size: int,
):

    # model init
    model = SentenceTransformer(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Data init
    ds = load_from_disk(ds_path)["train"]
    ds = ds.rename_columns(
        {"decompilation": "anchor", "source": "positive"}
    ).remove_columns("summary")  # may use this later

    half_length = int(
        round(model.max_seq_length / 2)
    )  # keeping the context limited for this example model

    # Used to truncate inputs into desired lengths.
    encode_decode = lambda x: tokenizer.decode(
        tokenizer.encode(x, truncation=True, max_length=half_length),
        skip_special_tokens=True,
    )
    ds = ds.map(
        lambda x: {
            "anchor": encode_decode(x["anchor"]),
            "positive": encode_decode(x["positive"]),
        }
    )  # not totally convinced the data is getting truncated, so we do it manually.

    ds = ds.select_columns(["anchor", "positive"])
    split_ds = ds.train_test_split(test_size=0.01)
    train_ds, val_ds = split_ds["train"], split_ds["test"]

    mini_batch_size = 1  # In nearly every case, this should always be 1
    eval_batch_size = max(int(batch_size / 16), 1)  # just to be safe

    # misc init
    loss = CachedMultipleNegativesRankingLoss(model, mini_batch_size=mini_batch_size)

    run_name = f"mitRE_embed_{model_name.replace('/', '_')}_test"
    out_path = Path(__file__).resolve().parent.parent / Path(
        f"models/mitRE_embed_{model_short_name}/{run_name}"
    )

    args = SentenceTransformerTrainingArguments(
        dataloader_drop_last=True,
        torch_empty_cache_steps=1,  # prevents OOM
        output_dir=str(out_path),
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
        eval_strategy="steps",
        eval_steps=10,
        save_strategy="steps",
        save_steps=10,
        save_total_limit=2,
        logging_steps=1,
        logging_first_step=True,
        run_name=run_name,
        report_to="tensorboard",
        skip_memory_metrics=False,
        save_safetensors=True,
    )

    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        loss=loss,
    )
    trainer.train()


if __name__ == "__main__":
    args = arg_parse()
    train(
        ds_path=args.data_dir_path,
        model_name=args.model_name,
        model_short_name=args.model_short_name,
        grad_accum_steps=args.grad_accum_steps,
        batch_size=args.batch_size,
    )
