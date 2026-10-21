"""Checkpoint evaluation must preserve the rebuttal's combined-pool ranking."""

import json

import numpy as np
import pytest
import torch


def test_ranks_match_rebuttal_with_ties_and_self_matches():
    from ifce.eval.checkpoint import combined_pool_metrics

    source = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    decompiled = torch.tensor([[0.8, 0.6], [1.0, 0.0], [0.0, 1.0]])
    pool = torch.cat([source, decompiled])
    expected = {}
    for name, queries, self_offset, target_offset in [
        ("decompiled_to_source", decompiled, 3, 0),
        ("source_to_decompiled", source, 0, 3),
    ]:
        ranks = []
        for i, query in enumerate(queries):
            scores = query @ pool.T
            target = scores[target_offset + i].clone()
            scores[self_offset + i] = -1e9
            ranks.append(int((scores > target).sum()) + 1)
        expected[name] = {
            "mrr": np.mean(1.0 / np.array(ranks)),
            "recall_at_1": np.mean(np.array(ranks) <= 1),
            "recall_at_10": np.mean(np.array(ranks) <= 10),
        }
    for chunk_size in (1, 2, 10):
        assert combined_pool_metrics(source, decompiled, chunk_size) == expected


@pytest.mark.parametrize(
    "bad", [torch.empty((0, 2)), torch.tensor([[float("nan"), 0]])]
)
def test_invalid_embeddings_are_rejected(bad):
    from ifce.eval.checkpoint import combined_pool_metrics

    with pytest.raises(ValueError):
        combined_pool_metrics(bad, bad)


def test_shape_mismatch_is_rejected():
    from ifce.eval.checkpoint import combined_pool_metrics

    with pytest.raises(ValueError):
        combined_pool_metrics(torch.ones(2, 3), torch.ones(3, 3))


@pytest.fixture(params=["bert", "mpnet"])
def checkpoint_and_dataset(tmp_path, request):
    """A real local Transformer checkpoint; no model download or mocking."""
    from datasets import Dataset, DatasetDict
    from sentence_transformers import SentenceTransformer, models
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import (
        BertConfig,
        BertModel,
        MPNetConfig,
        MPNetModel,
        PreTrainedTokenizerFast,
    )

    base = tmp_path / "base"
    tokenizer = Tokenizer(
        WordLevel({"[UNK]": 0, "[PAD]": 1, "add": 2, "mul": 3}, unk_token="[UNK]")
    )
    tokenizer.pre_tokenizer = Whitespace()
    PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, unk_token="[UNK]", pad_token="[PAD]"
    ).save_pretrained(base)
    config_class, model_class = (
        (MPNetConfig, MPNetModel)
        if request.param == "mpnet"
        else (BertConfig, BertModel)
    )
    model_class(
        config_class(
            vocab_size=4,
            hidden_size=8,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=16,
            max_position_embeddings=32,
        )
    ).save_pretrained(base)
    checkpoint = tmp_path / "checkpoint"
    SentenceTransformer(
        modules=[models.Transformer(str(base)), models.Pooling(8), models.Normalize()]
    ).save_pretrained(checkpoint)
    dataset = tmp_path / "data"
    DatasetDict(
        {
            "test": Dataset.from_dict(
                {"source": ["add", "mul"], "decompilation": ["add", "mul"]}
            )
        }
    ).save_to_disk(dataset)
    return checkpoint, dataset


def test_cli_evaluates_real_checkpoint_and_writes_json(
    checkpoint_and_dataset, tmp_path
):
    from typer.testing import CliRunner

    from ifce.cli.main import app

    checkpoint, dataset = checkpoint_and_dataset
    output = tmp_path / "results" / "metrics.json"
    result = CliRunner().invoke(
        app,
        [
            "eval",
            str(checkpoint),
            "--data-dir",
            str(dataset),
            "--device",
            "cpu",
            "--max-seq-length",
            "16",
            "--batch-size",
            "1",
            "--rank-batch-size",
            "1",
            "--output",
            str(output),
        ],
    )
    assert result.exit_code == 0, result.output + str(result.exception)
    report = json.loads(output.read_text())
    assert report["num_pairs"] == 2
    assert report["pool_size"] == 4
    assert report["split"] == "test"
    assert report["max_seq_length"] == 16
    assert report["checkpoint"] == str(checkpoint)
    assert report["metrics"]["decompiled_to_source"]["mrr"] == 1.0
    assert report["metrics"]["source_to_decompiled"]["recall_at_1"] == 1.0
    assert "Decompiled->Source" in result.output


def test_cli_uses_test_path_env(checkpoint_and_dataset, monkeypatch):
    from typer.testing import CliRunner

    from ifce.cli.main import app

    checkpoint, dataset = checkpoint_and_dataset
    monkeypatch.setenv("TEST_ASSEMBLAGE_PATH", str(dataset))
    result = CliRunner().invoke(
        app, ["eval", str(checkpoint), "--device", "cpu", "--max-seq-length", "16"]
    )
    assert result.exit_code == 0, result.output + str(result.exception)


def test_cli_requires_dataset_and_valid_batch_sizes(tmp_path, monkeypatch):
    from typer.testing import CliRunner

    from ifce.cli.main import app

    monkeypatch.delenv("TEST_ASSEMBLAGE_PATH", raising=False)
    runner = CliRunner()
    assert runner.invoke(app, ["eval", "checkpoint"]).exit_code == 2
    result = runner.invoke(
        app, ["eval", "checkpoint", "--data-dir", str(tmp_path), "--batch-size", "0"]
    )
    assert result.exit_code == 2


def test_ambiguous_datasetdict_requires_split(tmp_path):
    from datasets import Dataset, DatasetDict

    from ifce.eval.checkpoint import load_eval_dataset

    dataset = Dataset.from_dict({"source": ["x"], "decompilation": ["x"]})
    DatasetDict({"train": dataset, "validation": dataset}).save_to_disk(tmp_path)
    with pytest.raises(ValueError, match="split"):
        load_eval_dataset(tmp_path)
    selected, split = load_eval_dataset(tmp_path, "validation")
    assert len(selected) == 1 and split == "validation"


@pytest.mark.parametrize(
    "columns", [{"source": ["x"]}, {"source": [], "decompilation": []}]
)
def test_invalid_dataset_is_rejected(tmp_path, columns):
    from datasets import Dataset

    from ifce.eval.checkpoint import load_eval_dataset

    Dataset.from_dict(columns).save_to_disk(tmp_path, num_shards=1)
    with pytest.raises(ValueError):
        load_eval_dataset(tmp_path)
