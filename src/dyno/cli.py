"""Command-line interface for dyno."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dyno.dynare_model import DynareModel
from dyno.report import RunResults


def dynare(
    filename: str | Path,
    default_pipeline: bool = False,
    strict: bool = False,
) -> RunResults:
    """Import a Dynare .mod file and run it.

    Parameters
    ----------
    filename:
        Path to the .mod file.
    default_pipeline:
        When ``True`` and no run commands are defined in the model, runs the
        default pipeline: residuals, perturbation solve, and IRFs.
    strict:
        When ``False`` (default), the parser automatically declares parameters
        that are assigned values without being explicitly listed in the
        ``parameters`` section (``allow_undeclared_params=True``).  Set to
        ``True`` to enforce strict Dynare syntax and raise an error on
        undeclared parameters.

    Returns
    -------
    RunResults
        The results produced by the model's run commands.
    """
    model = DynareModel(
        filename=Path(filename),
        allow_undeclared_params=not strict,
    )
    return model.run(default_pipeline=default_pipeline)


def dynare_cmd(argv: list[str] | None = None) -> None:
    """Entry point for the ``dynare`` command."""
    parser = argparse.ArgumentParser(
        prog="dynare",
        description="Import and run a Dynare .mod file.",
    )
    parser.add_argument("filename", help="Path to the .mod file to run.")
    parser.add_argument(
        "--default-pipeline",
        action="store_true",
        default=False,
        help="Run the default pipeline (residuals, solve, IRFs) when no run commands are defined.",
    )
    args = parser.parse_args(argv)

    path = Path(args.filename)
    if not path.exists():
        print(f"Error: file not found: {path}", file=sys.stderr)
        sys.exit(1)

    results = dynare(path, default_pipeline=args.default_pipeline)
    print(results)


if __name__ == "__main__":
    dynare_cmd()
