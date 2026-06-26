# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The MITRE Corporation
"""Unit tests for the Signsrch signature-file parser."""

from ifce.eval.signsrch_benchmark import parse_signsrch

FIXTURE = """\
##########
TITLE
Test Algo One
TYPE
xor
DATA
0x12345678 0xdeadbeef
##########
TITLE
Test Algo Two
TYPE
add
DATA
0x00000001
"""


def test_parse_signsrch_extracts_titles_and_data(tmp_path):
    sig = tmp_path / "fixture.sig"
    sig.write_text(FIXTURE)

    result = parse_signsrch(str(sig))

    assert set(result) == {"Test Algo One", "Test Algo Two"}
    # hex constants are normalized to uppercase
    assert "0xDEADBEEF" in result["Test Algo One"]
    assert "0x12345678" in result["Test Algo One"]
    assert result["Test Algo Two"] == "0x1"


def test_parse_signsrch_skips_blocks_without_title_or_data(tmp_path):
    sig = tmp_path / "partial.sig"
    sig.write_text("##########\nTYPE\nxor\nDATA\n0x01\n")
    assert parse_signsrch(str(sig)) == {}
