# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The MITRE Corporation
"""Convenience entry point: run the full evaluation pipeline.

Equivalent to the ``ifce-experiment`` console script installed with the package.
"""

from ifce.cli.conduct_experiment import conduct_experiment


def main():
    conduct_experiment()


if __name__ == "__main__":
    main()
