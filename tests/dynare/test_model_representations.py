"""Render a model's text/HTML/markdown representations for every backend into
one combined HTML document, so they can be visually compared side by side.

The rendered document is written under ``tests/output/`` (git-ignored) and its
path is printed so it can be opened in a browser after the test run.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.render_model_representations import discover_models, render_report


def test_model_representations_report():
    output_path = render_report(
        [Path("examples/modfiles/RBC.mod")],
        Path("tests/output/model_representations.html"),
    )
    document = output_path.read_text()
    assert "DynoModel" in document
    assert "DynareModel" in document
    assert "examples/modfiles/RBC.mod" in document


def test_dyno_model_representations_report():
    output_path = render_report(
        [Path("examples/rbc.dyno")],
        Path("tests/output/dyno_model_representations.html"),
    )
    document = output_path.read_text()
    assert "DynoModel" in document
    assert "examples/rbc.dyno" in document


def test_discover_models_finds_both_supported_extensions():
    paths = discover_models(["examples"])
    suffixes = {path.suffix for path in paths}
    assert ".mod" in suffixes
    assert ".dyno" in suffixes
