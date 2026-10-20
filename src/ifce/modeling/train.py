# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The MITRE Corporation
import argparse

from datasets import load_from_disk
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.losses import (
    CachedMultipleNegativesRankingLoss,
)
from sentence_transformers.training_args import BatchSamplers
from transformers import AutoTokenizer, set_seed

from ifce.paths import CHECKPOINTS_DIR


def arg_parse():
    parser = argparse.ArgumentParser(
        description="Conduct Assemblage training.",
    )

    parser.add_argument(
        "--data_dir_path",
        required=True,
        type=str,
        help="Path to the Assemblage training dataset (a Hugging Face dataset saved with save_to_disk).",
    )

    parser.add_argument(
        "--seed",
        default=42,
        type=int,
        help="Random seed for reproducible data splitting and training.",
    )

    parser.add_argument(
        "--model_name",
        default="Qwen/Qwen3-Embedding-0.6B",
        type=str,
        help="Base embedding model to fine-tune.",
    )

    parser.add_argument(
        "--model_id",
        default="BUSTA-IFCE-Qwen3-0.6B",
        type=str,
        help="Name of the trained model; also the checkpoint directory name under "
        "outputs/checkpoints, which must match the id used in the experiment config.",
    )

    parser.add_argument(
        "--grad_accum_steps",
        default=2,
        type=int,
        help="Gradient accumulatioin steps. Should match the train config",
    )

    parser.add_argument("--batch_size", default=64, type=int, help="Batch size")

    parser.add_argument(
        "--max_seq_length",
        default=2048,
        type=int,
        help="Max input length in tokens. Caps the model context (and thus the "
        "per-side truncation of the training data). Qwen3-Embedding-0.6B defaults "
        "to 32768, which is slow and OOM-prone on long functions; 2048 matches the "
        "reported runs.",
    )

    parser.add_argument(
        "--loss_mini_batch_size",
        default=16,
        type=int,
        help="Chunk size for CachedMultipleNegativesRankingLoss. Affects speed and "
        "memory only, not the result (the cached gradient is identical). Larger is "
        "faster; lower it if you hit OOM.",
    )

    args = parser.parse_args()
    return args


def train(
    ds_path: str,
    model_name: str,
    model_id: str,
    grad_accum_steps: int,
    batch_size: int,
    seed: int = 42,
    max_seq_length: int = 2048,
    loss_mini_batch_size: int = 16,
):

    # Seed everything for reproducible data splits and training. Note that GPU /
    # cuDNN / DeepSpeed kernels remain a source of irreducible nondeterminism, so
    # exact bit-for-bit reproduction of result tables is not guaranteed.
    set_seed(seed)

    # model init
    model = SentenceTransformer(model_name)
    # Cap the context before deriving the data truncation below. Qwen3-Embedding-0.6B
    # defaults to a 32k context, which makes the rare very long functions slow and
    # OOM-prone; capping here matches the reported runs.
    model.max_seq_length = max_seq_length
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
    def encode_decode(x):
        return tokenizer.decode(
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
    split_ds = ds.train_test_split(test_size=0.01, seed=seed)
    train_ds, val_ds = split_ds["train"], split_ds["test"]

    eval_batch_size = max(int(batch_size / 16), 1)  # just to be safe

    # misc init. mini_batch_size only chunks the cached-loss embedding computation
    # (speed/memory); it does not change the gradient or the result.
    loss = CachedMultipleNegativesRankingLoss(
        model, mini_batch_size=loss_mini_batch_size
    )

    run_name = model_id
    out_path = CHECKPOINTS_DIR / model_id

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
        seed=seed,
        data_seed=seed,
    )

    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        loss=loss,
    )
    trainer.train()


def main():
    args = arg_parse()
    train(
        ds_path=args.data_dir_path,
        model_name=args.model_name,
        model_id=args.model_id,
        grad_accum_steps=args.grad_accum_steps,
        batch_size=args.batch_size,
        seed=args.seed,
        max_seq_length=args.max_seq_length,
        loss_mini_batch_size=args.loss_mini_batch_size,
    )


if __name__ == "__main__":
    main()
