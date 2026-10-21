# Checkpoint Evaluation Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans to implement these tasks.

**Goal:** Let collaborators evaluate an arbitrary checkpoint from main with one command.

**Architecture:** Add a Typer `ifce eval` command backed by an importable retrieval
evaluator. Preserve the combined-pool protocol of the standalone rebuttal script
and retain the existing paper benchmark commands.

**Tech Stack:** Python, Typer, SentenceTransformers, datasets, PyTorch, pytest.

**Spec:** User request in this chat: checkpoint evaluation on main, easy CLI, Typer.

## Global Constraints

- No internal hostnames or data paths in tracked sources.
- Checkpoint paths and Hub IDs work without registration or GPU-only extras.
- Default 2048-token cap; whole test split; normalized embeddings; both directions.
- Preserve strict-greater optimistic ties and self-match exclusion from rebuttal.

## Review Focus

- Tests pin tied scores and self-matches against the original ranking expression.
- Real offline checkpoint exercises loading, token cap, CLI and JSON end to end.
- Reject missing columns, empty datasets, and ambiguous splits.
- Keep all existing console scripts available.
- Chunk scoring rather than allocate a full 2N by 2N matrix.

## Tasks

- [x] Add regression tests for ranking parity, invalid inputs and local-checkpoint CLI.
- [x] Implement `ifce.eval.checkpoint` and the Typer entry point; update dependency lock.
- [x] Document one-command usage, environment override, settings and scoring caveats.
- [x] Run full tests/lint, review, and validate the VM CLI against a rebuttal checkpoint.

Validation: 64 tests pass; Ruff checks and wheel build pass. Independent review
confirmed the ranking protocol and identified forced SDPA as incompatible with
MPNet; the loader now selects the supported backend and BERT/MPNet checkpoints
both have offline CLI coverage. The original and new evaluators produced identical
printed metrics on an eight-pair VM sample using the saved Qwen seed42 checkpoint.

Delivery: labeled merge request to main, followed by the copyable evaluation command.
