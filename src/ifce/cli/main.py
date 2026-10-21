# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The MITRE Corporation
"""Typer entry point for direct checkpoint evaluation (``ifce eval``)."""

import json
from pathlib import Path
from typing import Annotated

import typer

app = typer.Typer(
    no_args_is_help=True, help="Train and evaluate identifier-free code embeddings."
)


@app.callback()
def main():
    """IFCE tools. Existing ifce-train, ifce-experiment and ifce-time remain available."""


@app.command("eval")
def eval_checkpoint(
    checkpoint: Annotated[
        str, typer.Argument(help="Local checkpoint directory or Hugging Face model ID.")
    ],
    data_dir: Annotated[
        Path,
        typer.Option(
            "--data-dir",
            envvar="TEST_ASSEMBLAGE_PATH",
            exists=True,
            file_okay=False,
            help="Test dataset saved with datasets.save_to_disk; or set TEST_ASSEMBLAGE_PATH.",
        ),
    ],
    device: Annotated[
        str | None,
        typer.Option(
            help="cuda, cuda:1, or cpu. Default: CUDA if available, otherwise CPU."
        ),
    ] = None,
    max_seq_length: Annotated[
        int,
        typer.Option(
            min=1, help="Maximum input tokens; match this across compared models."
        ),
    ] = 2048,
    batch_size: Annotated[
        int, typer.Option(min=1, help="Encoding batch size; reduce on CUDA OOM.")
    ] = 128,
    rank_batch_size: Annotated[
        int,
        typer.Option(
            min=1, help="Queries scored at once against the combined candidate pool."
        ),
    ] = 512,
    split: Annotated[
        str | None,
        typer.Option(help="DatasetDict split. Defaults to test, or the only split."),
    ] = None,
    output: Annotated[
        Path | None,
        typer.Option(help="Write metrics and evaluation settings to this JSON file."),
    ] = None,
    trust_remote_code: Annotated[
        bool,
        typer.Option(help="Allow model-supplied code (only for trusted checkpoints)."),
    ] = False,
):
    """Evaluate a checkpoint: combined-pool MRR, Recall@1/10 in both directions."""
    from ifce.eval.checkpoint import evaluate_checkpoint

    try:
        report = evaluate_checkpoint(
            checkpoint,
            data_dir,
            device=device,
            max_seq_length=max_seq_length,
            batch_size=batch_size,
            rank_batch_size=rank_batch_size,
            split=split,
            trust_remote_code=trust_remote_code,
        )
    except (ValueError, OSError) as exc:
        typer.echo(f"Evaluation failed: {exc}", err=True)
        raise typer.Exit(1) from exc
    typer.echo(
        f"EVAL {checkpoint} ({report['num_pairs']} pairs; {report['pool_size']} candidates; self-matches excluded)"
    )
    for key, name in (
        ("decompiled_to_source", "Decompiled->Source"),
        ("source_to_decompiled", "Source->Decompiled"),
    ):
        metrics = report["metrics"][key]
        typer.echo(
            f"  {name}: MRR={metrics['mrr']:.4f} R@1={metrics['recall_at_1']:.4f} R@10={metrics['recall_at_10']:.4f}"
        )
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2) + "\n")
        typer.echo(f"Results saved to {output}")


if __name__ == "__main__":
    app()
