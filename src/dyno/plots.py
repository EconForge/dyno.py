from __future__ import annotations

from typing import Any
import pandas as pd

from dyno.simul import (
    IRFSimulation,
    RandomSimulation,
    SimulationResult,
    TransitionSimulation,
    sim_to_nsim,
)
from dyno.typedefs import UnitsType
from dyno.variants import VariantCollection

ENGINES = ("altair", "plotext")


def _check_engine(engine: str) -> None:
    """Raise a helpful error for unsupported plotting engines."""
    if engine not in ENGINES:
        hint = (
            " Plotly support has been removed; use Altair instead."
            if engine == "plotly"
            else ""
        )
        raise ValueError(
            f"Unknown plotting engine {engine!r}; expected one of {ENGINES}.{hint}"
        )


def _prepare_nsim(
    sim: Any,
    *,
    variables: list[str] | None = None,
    T: int | None = None,
    units: UnitsType | None = None,
) -> pd.DataFrame:
    """Convert any simulation object into a filtered long-format DataFrame."""
    if isinstance(sim, SimulationResult):
        nsim = sim_to_nsim(sim, units=units).copy()
    elif isinstance(sim, dict):
        if not sim:
            return pd.DataFrame(columns=["shock", "t", "variable", "value"])
        nsim = sim_to_nsim(sim).copy()
    elif hasattr(sim, "melt"):
        df = sim.copy()
        if "t" not in df.columns:
            df = df.reset_index().rename(columns={"index": "t"})
        nsim = df.melt(id_vars=["t"], var_name="variable", value_name="value")
        nsim["shock"] = "simulation"
    else:
        return pd.DataFrame(columns=["shock", "t", "variable", "value"])

    if variables is not None:
        var_set = {str(v) for v in variables}
        nsim = nsim[nsim["variable"].astype(str).isin(var_set)]

    if T is not None:
        nsim = nsim[pd.to_numeric(nsim["t"], errors="coerce") <= int(T)]

    return nsim


def _prepare_variants_nsim(
    variants: VariantCollection[Any],
    *,
    variables: list[str] | None = None,
    shocks: list[str] | str | None = None,
    T: int | None = None,
    units: UnitsType | None = None,
) -> pd.DataFrame:
    """Convert a VariantCollection of simulations into a tidy DataFrame with a ``variant`` column."""
    frames: list[pd.DataFrame] = []
    for label, sim in zip(variants.labels, variants.items):
        sub = _prepare_nsim(sim, variables=variables, T=T, units=units)
        if sub.empty:
            continue
        sub = sub.copy()
        sub.insert(0, "variant", str(label))
        frames.append(sub)

    if not frames:
        return pd.DataFrame(columns=["variant", "shock", "t", "variable", "value"])

    nsim = pd.concat(frames, ignore_index=True)

    if shocks is not None:
        if isinstance(shocks, str):
            shock_set = {shocks}
        else:
            shock_set = {str(s) for s in shocks}
        nsim = nsim[nsim["shock"].astype(str).isin(shock_set)]

    return nsim


def plot_variants(
    variants: VariantCollection[Any],
    variables: list[str] | None = None,
    shocks: list[str] | str | None = None,
    T: int | None = None,
    units: UnitsType | None = None,
    engine: str = "altair",
    cols: int = 2,
    **kwargs: Any,
) -> Any:
    """Unify a ``VariantCollection`` of simulations into a single comparative plot.

    Parameters
    ----------
    variants : VariantCollection
        Collection of simulations.
    variables : list[str] | None, optional
        Subset of variable names to plot (also accepts ``vars`` in ``kwargs``).
    shocks : list[str] | str | None, optional
        Subset of shock names to include when plotting IRFs.
    T : int | None, optional
        Maximum time horizon to display.
    units : {'level', 'deviation', 'percent', 'log-deviation'} | None, optional
        Units to convert into before plotting.
    engine : {'altair', 'plotext'}, default 'altair'
        Plotting backend.
    cols : int, default 2
        Number of facet columns.
    """
    if variables is None and "vars" in kwargs:
        variables = kwargs.pop("vars")
    if units is None and "type" in kwargs:
        units = kwargs.pop("type")

    sim_vc = variants

    if engine == "plotext":
        return plot_variants_plotext(
            sim_vc,
            variables=variables,
            shocks=shocks,
            T=T,
            units=units,
            cols=cols,
            **kwargs,
        )
    _check_engine(engine)

    nsim = _prepare_variants_nsim(
        sim_vc,
        variables=variables,
        shocks=shocks,
        T=T,
        units=units,
    )

    is_spaghetti = any(
        isinstance(item, RandomSimulation) and item.N > 1 for item in sim_vc.items
    )
    unique_shocks = (
        list(dict.fromkeys(nsim["shock"].astype(str))) if not nsim.empty else []
    )
    is_multi_shock = (not is_spaghetti) and len(unique_shocks) > 1

    params = sim_vc.parameters
    legend_title = params[0] if len(params) == 1 else "Variant"
    variant_order = list(sim_vc.labels)

    import altair as alt

    if is_spaghetti:
        max_draws = max(
            (item.N for item in sim_vc.items if isinstance(item, RandomSimulation)),
            default=10,
        )
        opacity = max(0.15, min(0.6, 3.0 / max(1, max_draws)))
        nsim = nsim.copy()
        nsim["_trace_group"] = (
            nsim["variant"].astype(str) + "__" + nsim["shock"].astype(str)
        )
        ch = (
            alt.Chart(nsim)
            .mark_line(opacity=opacity)
            .encode(
                x="t:Q",
                y="value:Q",
                color=alt.Color("variant:N", sort=variant_order, title=legend_title),
                detail="_trace_group:N",
                facet=alt.Facet("variable:N", columns=cols),
            )
            .properties(width=220, height=120)
            .resolve_scale(y="independent")
            .interactive()
        )
        return ch
    elif is_multi_shock:
        ch = (
            alt.Chart(nsim)
            .mark_line()
            .encode(
                x="t:Q",
                y="value:Q",
                color=alt.Color("variant:N", sort=variant_order, title=legend_title),
                strokeDash=alt.StrokeDash("shock:N", title="Shock"),
                facet=alt.Facet("variable:N", columns=cols),
            )
            .properties(width=220, height=120)
            .resolve_scale(y="independent")
            .interactive()
        )
        return ch
    else:
        ch = (
            alt.Chart(nsim)
            .mark_line()
            .encode(
                x="t:Q",
                y="value:Q",
                color=alt.Color("variant:N", sort=variant_order, title=legend_title),
                facet=alt.Facet("variable:N", columns=cols),
            )
            .properties(width=220, height=120)
            .resolve_scale(y="independent")
            .interactive()
        )
        return ch


def plot_simulation(
    sim: Any,
    variables: list[str] | None = None,
    T: int | None = None,
    units: UnitsType | None = None,
    engine: str = "altair",
    **kwargs: Any,
) -> Any:
    """Plot any SimulationResult (IRF, Random spaghetti, or Transition), VariantCollection, dict, or DataFrame.

    Parameters
    ----------
    sim : SimulationResult | VariantCollection | dict | pd.DataFrame
        Simulation object or data to plot.
    variables : list[str] | None, optional
        Subset of variable names to plot (also accepts ``vars`` in ``kwargs``).
    T : int | None, optional
        Maximum time horizon to display.
    units : {'level', 'deviation', 'percent', 'log-deviation'} | None, optional
        Units to convert into before plotting.
    engine : {'altair', 'plotext'}, default 'altair'
        Plotting backend.
    """
    if variables is None and "vars" in kwargs:
        variables = kwargs.pop("vars")
    if units is None and "type" in kwargs:
        units = kwargs.pop("type")

    if isinstance(sim, VariantCollection):
        return plot_variants(
            sim,
            variables=variables,
            T=T,
            units=units,
            engine=engine,
            **kwargs,
        )

    if engine == "plotext":
        if isinstance(sim, SimulationResult) and units is not None:
            sim_data = sim.to_dict(units=units)
        else:
            sim_data = sim
        return plot_simulation_plotext(sim_data, variables=variables, **kwargs)
    _check_engine(engine)

    nsim = _prepare_nsim(sim, variables=variables, T=T, units=units)
    is_spaghetti = isinstance(sim, RandomSimulation) and sim.N > 1
    is_single = (
        isinstance(sim, TransitionSimulation)
        or (isinstance(sim, RandomSimulation) and sim.N == 1)
        or (not isinstance(sim, (SimulationResult, dict)) and hasattr(sim, "melt"))
    )

    import altair as alt

    if is_spaghetti:
        n_draws = sim.N if isinstance(sim, RandomSimulation) else 10
        opacity = max(0.15, min(0.6, 3.0 / max(1, n_draws)))
        ch = (
            alt.Chart(nsim)
            .mark_line(opacity=opacity)
            .encode(
                x="t:Q",
                y="value:Q",
                detail="shock:N",
                facet=alt.Facet("variable:N", columns=2),
            )
            .properties(width=200, height=100)
            .resolve_scale(y="independent")
            .interactive()
        )
        return ch
    elif is_single:
        ch = (
            alt.Chart(nsim)
            .mark_line()
            .encode(
                x="t:Q",
                y="value:Q",
                facet=alt.Facet("variable:N", columns=2),
            )
            .properties(width=200, height=100)
            .resolve_scale(y="independent")
            .interactive()
        )
        return ch
    else:
        ch = (
            alt.Chart(nsim)
            .mark_line()
            .encode(
                x="t:Q",
                y="value:Q",
                color="shock:N",
                facet=alt.Facet("variable:N", columns=2),
            )
            .properties(width=200, height=100)
            .resolve_scale(y="independent")
            .interactive()
        )
        return ch


def plot_irfs(sim: Any, engine: str = "altair", **kwargs: Any) -> Any:
    """Plot impulse response functions using Altair or Plotext.

    Parameters
    ----------
    sim : SimulationResult | VariantCollection | dict | pd.DataFrame
        Simulation or IRF data.
    engine : {"altair", "plotext"}, default "altair"
        Plotting engine to use.
    **kwargs : Any
        Additional arguments passed to the engine-specific plot function.
    """
    if isinstance(sim, VariantCollection):
        return plot_variants(sim, engine=engine, **kwargs)
    if isinstance(sim, SimulationResult) and not isinstance(sim, IRFSimulation):
        return plot_simulation(sim, engine=engine, **kwargs)

    if engine == "plotext":
        return plot_irfs_plotext(sim, **kwargs)
    _check_engine(engine)
    if isinstance(sim, dict):
        return plot_irfs_altair(sim)
    else:
        return plot_irf_altair(sim)


def plot_variants_plotext(
    variants: VariantCollection[Any],
    *,
    cols: int = 2,
    width: int | None = None,
    height: int | None = None,
    variables: list[str] | None = None,
    shocks: list[str] | str | None = None,
    T: int | None = None,
    units: UnitsType | None = None,
    color: bool | None = None,
    theme: str = "clear",
    marker: str | None = None,
    show: bool = False,
    **kwargs: Any,
) -> str:
    """Render multi-variant simulation trajectories as an ASCII/ANSI plot using plotext."""
    import shutil
    import sys
    import numpy as np

    if variables is None and "vars" in kwargs:
        variables = kwargs.pop("vars")
    if units is None and "type" in kwargs:
        units = kwargs.pop("type")

    try:
        import plotext as plt
    except ImportError:
        return "plotext is required for text graph rendering"

    sim_vc = variants
    data = _prepare_variants_nsim(
        sim_vc,
        variables=variables,
        shocks=shocks,
        T=T,
        units=units,
    )
    if data.empty:
        return ""

    data["t"] = pd.to_numeric(data["t"], errors="coerce")
    data["value"] = pd.to_numeric(data["value"], errors="coerce")
    data = data[np.isfinite(data["t"]) & np.isfinite(data["value"])]
    if data.empty:
        return ""

    selected_vars = list(dict.fromkeys(data["variable"].astype(str)))
    if not selected_vars:
        return ""

    unique_shocks = list(dict.fromkeys(data["shock"].astype(str)))
    variant_labels = list(dict.fromkeys(data["variant"].astype(str)))
    is_spaghetti = any(
        isinstance(item, RandomSimulation) and item.N > 1 for item in sim_vc.items
    )

    num_vars = len(selected_vars)
    actual_cols = max(1, min(cols, num_vars))
    rows = (num_vars + actual_cols - 1) // actual_cols

    if width is None:
        term_width = shutil.get_terminal_size(fallback=(80, 24)).columns
        width = max(40, min(term_width, 100))

    if height is None:
        height = max(8, rows * 10)

    plt.main()
    plt.clf()
    if theme:
        try:
            plt.theme(theme)
        except Exception:
            pass
    plt.plotsize(width, height)
    plt.subplots(rows, actual_cols)

    for i, var in enumerate(selected_vars):
        r = i // actual_cols + 1
        c = i % actual_cols + 1
        plt.subplot(r, c)
        plt.title(str(var))
        var_df = data[data["variable"].astype(str) == var]
        for v_lbl in variant_labels:
            vdf = var_df[var_df["variant"].astype(str) == v_lbl]
            for s_idx, shock in enumerate(unique_shocks):
                sdf = vdf[vdf["shock"].astype(str) == shock].sort_values("t")
                if sdf.empty:
                    continue
                if is_spaghetti:
                    lbl = v_lbl if s_idx == 0 else None
                elif len(unique_shocks) > 1:
                    lbl = f"{v_lbl} ({shock})"
                else:
                    lbl = v_lbl
                plot_kwargs: dict[str, Any] = {"label": lbl}
                if marker is not None:
                    plot_kwargs["marker"] = marker
                plt.plot(list(sdf["t"]), list(sdf["value"]), **plot_kwargs)

    for i in range(num_vars, rows * actual_cols):
        r = i // actual_cols + 1
        c = i % actual_cols + 1
        plt.subplot(r, c)
        plt.frame(False)
        plt.xticks([])
        plt.yticks([])

    output = plt.build()

    if color is None:
        color = bool(hasattr(sys.stdout, "isatty") and sys.stdout.isatty())
    if not color:
        output = plt.uncolorize(output)

    if show:
        print(output)

    return output


def plot_simulation_plotext(
    sim: Any,
    *,
    cols: int = 2,
    width: int | None = None,
    height: int | None = None,
    variables: list[str] | None = None,
    color: bool | None = None,
    theme: str = "clear",
    marker: str | None = None,
    show: bool = False,
) -> str:
    """Render simulation or IRF paths as a text-based ASCII/ANSI plot using plotext."""
    import shutil
    import sys
    import numpy as np

    try:
        import plotext as plt
    except ImportError:
        return "plotext is required for text graph rendering"

    if sim is None:
        return ""

    if isinstance(sim, SimulationResult):
        data = sim_to_nsim(sim).copy()
    elif isinstance(sim, dict):
        if not sim:
            return ""
        data = sim_to_nsim(sim).copy()
    elif hasattr(sim, "melt"):
        data = sim.copy()
        if "t" not in data.columns:
            data = data.reset_index().rename(columns={"index": "t"})
        data = data.melt(id_vars=["t"], var_name="variable", value_name="value")
        data["shock"] = "simulation"
    else:
        return ""

    required = {"t", "variable", "value", "shock"}
    if not required.issubset(data.columns):
        return ""

    data = data.loc[:, ["t", "variable", "value", "shock"]].copy()
    data["t"] = pd.to_numeric(data["t"], errors="coerce")
    data["value"] = pd.to_numeric(data["value"], errors="coerce")
    data = data[np.isfinite(data["t"]) & np.isfinite(data["value"])]
    if data.empty:
        return ""

    all_variables = list(dict.fromkeys(data["variable"].astype(str)))
    if variables is not None:
        var_set = set(variables)
        selected_vars = [v for v in all_variables if v in var_set]
    else:
        selected_vars = all_variables

    if not selected_vars:
        return ""

    shocks = list(dict.fromkeys(data["shock"].astype(str)))
    num_vars = len(selected_vars)
    actual_cols = max(1, min(cols, num_vars))
    rows = (num_vars + actual_cols - 1) // actual_cols

    if width is None:
        term_width = shutil.get_terminal_size(fallback=(80, 24)).columns
        width = max(40, min(term_width, 100))

    if height is None:
        height = max(8, rows * 10)

    # Master figure reset
    plt.main()
    plt.clf()
    if theme:
        try:
            plt.theme(theme)
        except Exception:
            pass
    plt.plotsize(width, height)
    plt.subplots(rows, actual_cols)

    is_spaghetti = isinstance(sim, RandomSimulation) and sim.N > 1

    for i, var in enumerate(selected_vars):
        r = i // actual_cols + 1
        c = i % actual_cols + 1
        plt.subplot(r, c)
        plt.title(str(var))
        var_df = data[data["variable"].astype(str) == var]
        for shock in shocks:
            sdf = var_df[var_df["shock"].astype(str) == shock].sort_values("t")
            if sdf.empty:
                continue
            lbl = str(shock) if (len(shocks) > 1 and not is_spaghetti) else None
            plot_kwargs: dict[str, Any] = {"label": lbl}
            if marker is not None:
                plot_kwargs["marker"] = marker
            plt.plot(list(sdf["t"]), list(sdf["value"]), **plot_kwargs)

    # Blank out any remaining unused cells in the grid
    for i in range(num_vars, rows * actual_cols):
        r = i // actual_cols + 1
        c = i % actual_cols + 1
        plt.subplot(r, c)
        plt.frame(False)
        plt.xticks([])
        plt.yticks([])

    output = plt.build()

    # Determine whether to retain ANSI color codes
    if color is None:
        color = bool(hasattr(sys.stdout, "isatty") and sys.stdout.isatty())
    if not color:
        output = plt.uncolorize(output)

    if show:
        print(output)

    return output


def plot_irfs_plotext(sim: Any, **kwargs: Any) -> str:
    """Render IRF paths as a text-based ASCII/ANSI plot using plotext."""
    return plot_simulation_plotext(sim, **kwargs)


def plot_irf_plotext(sim: Any, **kwargs: Any) -> str:
    """Render a single IRF path as a text-based ASCII/ANSI plot using plotext."""
    return plot_simulation_plotext(sim, **kwargs)


def plot_irf_altair(sim):
    import altair as alt

    if isinstance(sim, SimulationResult):
        sim = sim.to_df()
    else:
        sim = sim.copy()
    if "t" not in sim.columns:
        sim["t"] = sim.index
    sim = sim.melt(id_vars=["t"])

    ch = (
        alt.Chart(sim)
        .mark_line()
        .encode(x="t", y="value", facet=alt.Facet("variable", columns=2))
        .properties(width=200, height=100)
        .resolve_scale(y="independent")
        .interactive()
    )
    return ch


def plot_irfs_altair(sim):
    import altair as alt

    sim = sim_to_nsim(sim)

    ch = (
        alt.Chart(sim)
        .mark_line()
        .encode(x="t", y="value", color="shock", facet=alt.Facet("variable", columns=2))
        .properties(width=200, height=100)
        .resolve_scale(y="independent")
        .interactive()
    )
    return ch
