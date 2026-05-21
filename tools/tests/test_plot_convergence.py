"""Smoke test for tools/plot_convergence.py.

The script is non-interactive (writes a file when --output is given),
which is what we test here. We don't validate the actual plot content
beyond "the output file exists and is non-trivial in size".

Skipped when matplotlib isn't installed.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("matplotlib")

REPO = Path(__file__).resolve().parent.parent.parent
PLOT_TOOL = REPO / "tools" / "plot_convergence.py"


SAMPLE_CONV = """\
iteration,best_fitness,wall_seconds
-1,42.0,0
0,12.5,0.1
1,5.3,0.2
2,1.6,0.3
3,1.6,0.4
4,1.6,0.5
"""


def test_plot_writes_png(tmp_path):
    conv = tmp_path / "conv.csv"
    conv.write_text(SAMPLE_CONV)
    out = tmp_path / "plot.png"
    rc = subprocess.run(
        [sys.executable, str(PLOT_TOOL), str(conv), "--output", str(out)],
        capture_output=True, text=True, timeout=30,
    ).returncode
    assert rc == 0
    assert out.exists()
    assert out.stat().st_size > 1000  # not a blank file


def test_plot_writes_pdf_publication_style(tmp_path):
    conv = tmp_path / "conv.csv"
    conv.write_text(SAMPLE_CONV)
    out = tmp_path / "plot.pdf"
    rc = subprocess.run(
        [sys.executable, str(PLOT_TOOL), str(conv),
         "--output", str(out), "--style", "publication",
         "--mark-baseline", "--logy"],
        capture_output=True, text=True, timeout=30,
    ).returncode
    assert rc == 0
    assert out.exists()
    # PDF magic header
    with out.open("rb") as f:
        assert f.read(4) == b"%PDF"


def test_plot_multi_log_with_labels(tmp_path):
    """Two CSVs + --labels should produce a plot with both series."""
    c1 = tmp_path / "a.csv"
    c2 = tmp_path / "b.csv"
    c1.write_text(SAMPLE_CONV)
    c2.write_text(SAMPLE_CONV.replace("42.0", "50.0"))
    out = tmp_path / "compare.png"
    rc = subprocess.run(
        [sys.executable, str(PLOT_TOOL), str(c1), str(c2),
         "--labels", "uniform", "targeted", "--output", str(out)],
        capture_output=True, text=True, timeout=30,
    ).returncode
    assert rc == 0
    assert out.exists()


def test_plot_label_count_mismatch_errors(tmp_path):
    conv = tmp_path / "conv.csv"
    conv.write_text(SAMPLE_CONV)
    res = subprocess.run(
        [sys.executable, str(PLOT_TOOL), str(conv),
         "--labels", "one", "two", "three",  # 3 labels, 1 log
         "--output", str(tmp_path / "out.png")],
        capture_output=True, text=True, timeout=10,
    )
    assert res.returncode != 0
    assert "labels" in res.stderr.lower()
