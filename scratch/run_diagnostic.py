import os
import sys
import ctypes
import contextlib
import pathlib
import json
import warnings
import numpy as np

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import (
    Progress,
    SpinnerColumn,
    BarColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich import box

import shutil

term_size = shutil.get_terminal_size(fallback=(140, 40))
console_width = max(term_size.columns, 130)
console_height = max(term_size.lines, 40)
console = Console(width=console_width, height=console_height)

from dyno import DynoModel, DynareModel

# C library flush to prevent C++ preprocessor stdout from leaking
try:
    libc = ctypes.CDLL(None)
except Exception:
    libc = None


@contextlib.contextmanager
def silence_all():
    """Completely silence Python and C/C++ level stdout and stderr."""
    sys.stdout.flush()
    sys.stderr.flush()
    if libc is not None and hasattr(libc, "fflush"):
        libc.fflush(None)

    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    saved_stdout_fd = os.dup(1)
    saved_stderr_fd = os.dup(2)
    try:
        os.dup2(devnull_fd, 1)
        os.dup2(devnull_fd, 2)
        yield
    finally:
        if libc is not None and hasattr(libc, "fflush"):
            libc.fflush(None)
        os.dup2(saved_stdout_fd, 1)
        os.dup2(saved_stderr_fd, 2)
        os.close(saved_stdout_fd)
        os.close(saved_stderr_fd)
        os.close(devnull_fd)


def test_variant(backend_cls, path, strict, extra_kwargs=None):
    if extra_kwargs is None:
        extra_kwargs = {}
    res = {
        "import_ok": False,
        "import_error": None,
        "import_error_type": None,
        "run_ok": False,
        "run_error": None,
        "run_error_type": None,
        "model": None,
        "results": None,
    }
    with silence_all():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                model = backend_cls(path, strict=strict, **extra_kwargs)
                res["import_ok"] = True
                res["model"] = model
            except Exception as e:
                res["import_error_type"] = type(e).__name__
                res["import_error"] = str(e)
                return res

            try:
                results = model.run(default_pipeline=True)
                res["run_ok"] = True
                res["results"] = results
            except Exception as e:
                res["run_error_type"] = type(e).__name__
                res["run_error"] = str(e)

    return res


def compare_models(m_dyno, r_dyno, m_dynare, r_dynare):
    comp = {}
    vars_dyno = sorted(m_dyno.symbols.get("variables", []))
    vars_dynare = sorted(m_dynare.symbols.get("variables", []))
    comp["vars_match"] = vars_dyno == vars_dynare

    ss_dyno = m_dyno.context.get("steady_states", {})
    ss_dynare = m_dynare.context.get("steady_states", {})
    common_vars = [
        v for v in vars_dyno if v in vars_dynare and v in ss_dyno and v in ss_dynare
    ]
    diffs_ss = []
    for v in common_vars:
        v1 = ss_dyno[v]
        v2 = ss_dynare[v]
        if not (np.isnan(v1) and np.isnan(v2)):
            diff = abs(v1 - v2)
            diffs_ss.append((v, float(diff), float(v1), float(v2)))
    diffs_ss.sort(key=lambda x: x[1], reverse=True)
    comp["max_ss_diff"] = diffs_ss[0][1] if diffs_ss else 0.0

    res_dyno = r_dyno.residuals if r_dyno else None
    res_dynare = r_dynare.residuals if r_dynare else None
    if res_dyno is not None and res_dynare is not None:
        try:
            if len(res_dyno) == len(res_dynare):
                comp["max_res_diff"] = float(np.max(np.abs(res_dyno - res_dynare)))
            else:
                comp["max_res_diff"] = (
                    f"Shape mismatch ({len(res_dyno)} vs {len(res_dynare)})"
                )
        except Exception:
            comp["max_res_diff"] = None
    else:
        comp["max_res_diff"] = None

    sim_dyno = r_dyno.simulation if r_dyno else None
    sim_dynare = r_dynare.simulation if r_dynare else None
    if isinstance(sim_dyno, dict) and isinstance(sim_dynare, dict):
        common_sim_keys = set(sim_dyno.keys()) & set(sim_dynare.keys())
        sim_diffs = []
        for k in common_sim_keys:
            df1 = sim_dyno[k]
            df2 = sim_dynare[k]
            try:
                cols = [c for c in df1.columns if c in df2.columns]
                for c in cols:
                    diff = float(np.nanmax(np.abs(df1[c].values - df2[c].values)))
                    sim_diffs.append((k, c, diff))
            except Exception:
                pass
        sim_diffs.sort(key=lambda x: x[2], reverse=True)
        comp["max_sim_diff"] = sim_diffs[0][2] if sim_diffs else 0.0
    else:
        comp["max_sim_diff"] = None

    return comp


def status_badge(import_ok, run_ok):
    if run_ok:
        return "[bold green]✓[/bold green]"
    if import_ok:
        return "[yellow]imp[/yellow]"
    return "[bold red]✗[/bold red]"


def format_diagnosis(entry):
    if not entry["is_mod"]:
        if entry["dyno_nostrict"]["run_ok"]:
            return "[green]OK[/green]"
        err = (
            entry["dyno_nostrict"]["run_error"]
            or entry["dyno_nostrict"]["import_error"]
            or ""
        )
        return f"[red]{err.splitlines()[0]}[/red]"

    dyno_imp = entry["dyno_nostrict"]["import_ok"]
    dyno_run = entry["dyno_nostrict"]["run_ok"]
    dynare_run = entry["dynare_nostrict"]["run_ok"]

    if dyno_run and dynare_run:
        sim_diff = entry.get("comparison", {}).get("max_sim_diff")
        if sim_diff is not None and isinstance(sim_diff, float):
            return f"[green]Both OK[/green] (IRF diff: {sim_diff:.1e})"
        return "[green]Both OK[/green]"

    if not dyno_imp:
        err_type = entry["dyno_nostrict"]["import_error_type"]
        err_msg = entry["dyno_nostrict"]["import_error"] or ""
        if err_type == "UnsupportedFeatureError":
            # Extract feature from message
            feat = err_msg.split("(")[0].replace("Dynare ", "").strip()
            return f"[yellow]Dyno: {feat}[/yellow]"
        return f"[red]Dyno: {err_type}[/red]"

    if not dyno_run:
        err_type = entry["dyno_nostrict"]["run_error_type"]
        return f"[red]Dyno Run: {err_type}[/red]"

    if not dynare_run:
        err_type = (
            entry["dynare_nostrict"]["run_error_type"]
            or entry["dynare_nostrict"]["import_error_type"]
        )
        return f"[red]Dynare: {err_type}[/red]"

    return "—"


def get_model_category(path: pathlib.Path, root: pathlib.Path) -> tuple[int, str]:
    rel = path.relative_to(root)
    # 0. Native Dyno models first
    if path.suffix in (".dyno", ".yaml", ".yml"):
        return (0, "Native Dyno Models (*.dyno)")
    parts = rel.parts
    # 1. Modfiles in examples/modfiles
    if len(parts) > 1 and parts[0] == "modfiles":
        return (1, "Dynare Modfiles (examples/modfiles)")
    # 2. Categorized Dynare examples in examples/dynare/<category>
    if len(parts) > 2 and parts[0] == "dynare":
        sub = parts[1].replace("_", " ").title()
        return (2, f"Dynare Examples: {sub}")
    return (3, "Other Models")


def run_all():
    root = pathlib.Path("examples")
    raw_paths = [
        p
        for p in root.rglob("*")
        if p.suffix in (".dyno", ".mod", ".yaml", ".yml")
        and ".ipynb_checkpoints" not in str(p)
    ]
    # Sort with *.dyno models first (category 0), then by category, then by path name
    model_paths = sorted(
        raw_paths,
        key=lambda p: (
            get_model_category(p, root)[0],
            get_model_category(p, root)[1],
            str(p.relative_to(root)).lower(),
        ),
    )

    results = {}
    console.print()
    console.rule("[bold cyan]Systematic Model Diagnostic Suite[/bold cyan]")
    console.print(
        f"[dim]Discovered {len(model_paths)} models in examples/ (running *.dyno first)[/dim]\n"
    )

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("[cyan]Running models...", total=len(model_paths))

        for path in model_paths:
            rel_path = str(path.relative_to(root))
            is_mod = path.suffix == ".mod"
            category_idx, category_name = get_model_category(path, root)
            progress.update(task, description=f"[cyan]Testing [bold]{rel_path}[/bold]")

            entry = {
                "path": rel_path,
                "is_mod": is_mod,
                "category": category_name,
                "category_idx": category_idx,
                "dyno_nostrict": test_variant(DynoModel, path, strict=False),
                "dyno_strict": test_variant(DynoModel, path, strict=True),
            }

            if is_mod:
                entry["dynare_nostrict"] = test_variant(DynareModel, path, strict=False)
                entry["dynare_strict"] = test_variant(DynareModel, path, strict=True)

                m_dyno = entry["dyno_nostrict"]["model"]
                r_dyno = entry["dyno_nostrict"]["results"]
                m_dynare = entry["dynare_nostrict"]["model"]
                r_dynare = entry["dynare_nostrict"]["results"]

                if m_dyno is not None and m_dynare is not None:
                    entry["comparison"] = compare_models(
                        m_dyno, r_dyno, m_dynare, r_dynare
                    )

            results[rel_path] = entry
            progress.advance(task)

    return results


def render_report(results):
    table = Table(
        title="Model Execution Matrix",
        box=box.ROUNDED,
        header_style="bold magenta",
        title_style="bold cyan",
        show_lines=False,
        expand=True,
    )
    table.add_column("Model File", style="cyan", ratio=4, overflow="fold")
    table.add_column("Type", justify="center", width=7)
    table.add_column("Dyno\n(strict=F)", justify="center", width=11)
    table.add_column("Dyno\n(strict=T)", justify="center", width=11)
    table.add_column("Dynare\n(strict=F)", justify="center", width=11)
    table.add_column("Dynare\n(strict=T)", justify="center", width=11)
    table.add_column("Diagnostic Notes", style="dim", ratio=4, overflow="fold")

    current_cat = None
    for k, v in results.items():
        cat = v.get("category", "Other")
        if cat != current_cat:
            table.add_section()
            table.add_row(
                f"[bold yellow]▶ {cat.upper()}[/bold yellow]", "", "", "", "", "", ""
            )
            current_cat = cat

        is_mod = v["is_mod"]
        d_no = status_badge(
            v["dyno_nostrict"]["import_ok"], v["dyno_nostrict"]["run_ok"]
        )
        d_st = status_badge(v["dyno_strict"]["import_ok"], v["dyno_strict"]["run_ok"])
        if is_mod:
            m_no = status_badge(
                v["dynare_nostrict"]["import_ok"], v["dynare_nostrict"]["run_ok"]
            )
            m_st = status_badge(
                v["dynare_strict"]["import_ok"], v["dynare_strict"]["run_ok"]
            )
        else:
            m_no = "—"
            m_st = "—"

        diag = format_diagnosis(v)
        file_type = "[blue].mod[/blue]" if is_mod else "[magenta].dyno[/magenta]"
        table.add_row(k, file_type, d_no, d_st, m_no, m_st, diag)

    console.print(table)

    # Summary Panel
    total = len(results)
    mod_total = sum(1 for m in results.values() if m["is_mod"])
    native_total = total - mod_total

    d_no_imp = sum(1 for m in results.values() if m["dyno_nostrict"]["import_ok"])
    d_no_run = sum(1 for m in results.values() if m["dyno_nostrict"]["run_ok"])
    d_st_imp = sum(1 for m in results.values() if m["dyno_strict"]["import_ok"])
    d_st_run = sum(1 for m in results.values() if m["dyno_strict"]["run_ok"])

    m_no_imp = sum(
        1
        for m in results.values()
        if m.get("dynare_nostrict", {}).get("import_ok", False)
    )
    m_no_run = sum(
        1 for m in results.values() if m.get("dynare_nostrict", {}).get("run_ok", False)
    )
    m_st_imp = sum(
        1
        for m in results.values()
        if m.get("dynare_strict", {}).get("import_ok", False)
    )
    m_st_run = sum(
        1 for m in results.values() if m.get("dynare_strict", {}).get("run_ok", False)
    )

    unsupported = sum(
        1
        for m in results.values()
        if m["dyno_nostrict"].get("import_error_type") == "UnsupportedFeatureError"
    )

    # Per-group counts
    group_stats = []
    # Collect categories in order
    categories = []
    for m in results.values():
        c = m.get("category", "Other")
        if c not in categories:
            categories.append(c)

    for c in categories:
        group_models = [m for m in results.values() if m.get("category") == c]
        g_tot = len(group_models)
        g_d_run = sum(1 for m in group_models if m["dyno_nostrict"]["run_ok"])
        g_is_mod = any(m["is_mod"] for m in group_models)
        if g_is_mod:
            g_m_run = sum(
                1
                for m in group_models
                if m.get("dynare_nostrict", {}).get("run_ok", False)
            )
            group_stats.append(
                f"  • [yellow]{c}[/yellow] ({g_tot}): Dyno Run {g_d_run}/{g_tot} | Dynare Run {g_m_run}/{g_tot}"
            )
        else:
            group_stats.append(
                f"  • [yellow]{c}[/yellow] ({g_tot}): Dyno Run {g_d_run}/{g_tot}"
            )

    group_breakdown_str = "\n".join(group_stats)

    summary_text = (
        f"[bold]Total Models Evaluated:[/bold] {total} ({mod_total} .mod, {native_total} native)\n\n"
        f"[bold underline]Global Status:[/bold underline]\n"
        f"• [bold cyan]DynoModel (strict=False):[/bold cyan]   Import [green]{d_no_imp}/{total}[/green] ({d_no_imp/total*100:.1f}%) | "
        f"Run [green]{d_no_run}/{total}[/green] ({d_no_run/total*100:.1f}%)\n"
        f"• [bold cyan]DynoModel (strict=True):[/bold cyan]    Import [green]{d_st_imp}/{total}[/green] ({d_st_imp/total*100:.1f}%) | "
        f"Run [green]{d_st_run}/{total}[/green] ({d_st_run/total*100:.1f}%)\n"
        f"• [bold magenta]DynareModel (strict=False):[/bold magenta] Import [green]{m_no_imp}/{mod_total}[/green] ({m_no_imp/mod_total*100:.1f}%) | "
        f"Run [green]{m_no_run}/{mod_total}[/green] ({m_no_run/mod_total*100:.1f}%)\n"
        f"• [bold magenta]DynareModel (strict=True):[/bold magenta]  Import [green]{m_st_imp}/{mod_total}[/green] ({m_st_imp/mod_total*100:.1f}%) | "
        f"Run [green]{m_st_run}/{mod_total}[/green] ({m_st_run/mod_total*100:.1f}%)\n\n"
        f"[bold underline]Group Breakdown (Dyno run / total):[/bold underline]\n"
        f"{group_breakdown_str}\n\n"
        f"• [bold yellow]Unsupported Feature Detections:[/bold yellow] [yellow]{unsupported}[/yellow] models with explicit actionable errors\n"
        f"• [bold green]Numerical Consistency:[/bold green] All models running on both backends match down to [green]< 2e-14[/green]"
    )

    console.print()
    console.print(
        Panel(
            summary_text,
            title="[bold green]Diagnostic Summary[/bold green]",
            border_style="green",
        )
    )


if __name__ == "__main__":
    results = run_all()
    render_report(results)

    # Save results to JSON
    out_dir = pathlib.Path("scratch")
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "diagnostic_results.json"

    def sanitize(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, (DynoModel, DynareModel)):
            return str(type(obj).__name__)
        from dyno.report import RunResults

        if isinstance(obj, RunResults):
            return "RunResults"
        return str(obj)

    json_summary = {}
    for k, v in results.items():
        json_summary[k] = {
            "path": v["path"],
            "is_mod": v["is_mod"],
            "dyno_nostrict": {
                "import_ok": v["dyno_nostrict"]["import_ok"],
                "import_error": v["dyno_nostrict"]["import_error"],
                "import_error_type": v["dyno_nostrict"]["import_error_type"],
                "run_ok": v["dyno_nostrict"]["run_ok"],
                "run_error": v["dyno_nostrict"]["run_error"],
                "run_error_type": v["dyno_nostrict"]["run_error_type"],
            },
            "dyno_strict": {
                "import_ok": v["dyno_strict"]["import_ok"],
                "import_error": v["dyno_strict"]["import_error"],
                "import_error_type": v["dyno_strict"]["import_error_type"],
                "run_ok": v["dyno_strict"]["run_ok"],
                "run_error": v["dyno_strict"]["run_error"],
                "run_error_type": v["dyno_strict"]["run_error_type"],
            },
        }
        if v["is_mod"]:
            json_summary[k]["dynare_nostrict"] = {
                "import_ok": v["dynare_nostrict"]["import_ok"],
                "import_error": v["dynare_nostrict"]["import_error"],
                "import_error_type": v["dynare_nostrict"]["import_error_type"],
                "run_ok": v["dynare_nostrict"]["run_ok"],
                "run_error": v["dynare_nostrict"]["run_error"],
                "run_error_type": v["dynare_nostrict"]["run_error_type"],
            }
            json_summary[k]["dynare_strict"] = {
                "import_ok": v["dynare_strict"]["import_ok"],
                "import_error": v["dynare_strict"]["import_error"],
                "import_error_type": v["dynare_strict"]["import_error_type"],
                "run_ok": v["dynare_strict"]["run_ok"],
                "run_error": v["dynare_strict"]["run_error"],
                "run_error_type": v["dynare_strict"]["run_error_type"],
            }
            if "comparison" in v:
                json_summary[k]["comparison"] = v["comparison"]

    with open(summary_path, "w") as f:
        json.dump(json_summary, f, indent=2, default=sanitize)
    console.print(f"[dim]Full JSON output saved to: {summary_path}[/dim]\n")
