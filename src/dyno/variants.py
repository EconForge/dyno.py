from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping, Sequence
import html
import itertools
import time
from typing import Any, Callable, ClassVar, Generic, TypeVar, overload

import numpy as np
import pandas as pd

from .model import AbstractModel
from .report import RunResults
from .simul import SimulationResult
from .solver import PerturbationSolution, RecursiveDecisionRule
from .typedefs import IRFType, UnitsType

T = TypeVar("T")


def _format_scalar(val: Any) -> str:
    if isinstance(val, float):
        return f"{val:g}"
    return str(val)


def _format_variant_label(spec: Mapping[str, Any], index: int) -> str:
    if not spec:
        return f"variant_{index}"
    return ", ".join(f"{k}={_format_scalar(v)}" for k, v in spec.items())


def _is_sequence_like(val: Any) -> bool:
    if isinstance(val, (str, bytes, dict, Mapping)):
        return False
    return isinstance(val, (Sequence, np.ndarray, Iterable))


def _expand_variant_specs(
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[str]]:
    """Normalize variant arguments into a list of calibration dicts and labels.

    Supports:
    - ``variants(a=[2, 3, 4])`` -> Cartesian product of iterable keyword arguments
    - ``variants(a=[2, 3], b=[10, 20], _zip=True)`` -> pairwise zip of keyword arguments
    - ``variants([{"a": 2}, {"a": 3}])`` -> explicit list of calibration dicts
    - ``variants({"low": {"a": 2}, "high": {"a": 4}})`` -> named variant mapping
    """
    use_zip = bool(kwargs.pop("_zip", False))
    explicit_labels: list[str] | None = kwargs.pop("_labels", None)

    specs: list[dict[str, Any]] = []
    labels: list[str] = []

    if len(args) > 1:
        raise TypeError(
            f"variants() takes at most 1 positional argument ({len(args)} given)"
        )

    if len(args) == 1:
        arg = args[0]
        if isinstance(arg, Mapping):
            for lbl, calib in arg.items():
                if not isinstance(calib, Mapping):
                    raise TypeError(
                        f"Expected mapping of calibration values for variant {lbl!r}, got {type(calib).__name__}"
                    )
                merged = dict(calib) | kwargs
                specs.append(merged)
                labels.append(str(lbl))
            return specs, labels
        elif _is_sequence_like(arg):
            for idx, item in enumerate(arg):
                if not isinstance(item, Mapping):
                    raise TypeError(
                        f"Expected calibration dict at index {idx}, got {type(item).__name__}"
                    )
                merged = dict(item) | kwargs
                specs.append(merged)
                labels.append(_format_variant_label(merged, idx))
            if explicit_labels is not None:
                if len(explicit_labels) != len(specs):
                    raise ValueError(
                        f"Length of _labels ({len(explicit_labels)}) does not match number of variants ({len(specs)})"
                    )
                labels = [str(x) for x in explicit_labels]
            return specs, labels
        else:
            raise TypeError(
                "Positional argument to variants() must be a sequence of calibration dicts or a mapping of {label: calib_dict}"
            )

    if not kwargs:
        raise ValueError(
            "variants() requires at least one parameter specification (e.g. model.variants(a=[2, 3, 4]))."
        )

    param_names = list(kwargs.keys())
    param_values: list[list[Any]] = []
    for k in param_names:
        v = kwargs[k]
        if _is_sequence_like(v):
            vals = list(v)
            if len(vals) == 0:
                raise ValueError(f"Parameter {k!r} in variants() cannot be empty.")
            param_values.append(vals)
        else:
            param_values.append([v])

    if use_zip:
        lengths = {len(vals) for vals in param_values if len(vals) > 1}
        if len(lengths) > 1:
            raise ValueError(
                "All non-scalar parameter sequences must have the same length when _zip=True."
            )
        n_variants = max((len(vals) for vals in param_values), default=1)
        combos = [
            tuple(vals[i] if len(vals) > 1 else vals[0] for vals in param_values)
            for i in range(n_variants)
        ]
    else:
        combos = list(itertools.product(*param_values))

    varying_keys = [
        k for idx, k in enumerate(param_names) if len(param_values[idx]) > 1
    ]
    if not varying_keys:
        varying_keys = param_names

    for idx, combo in enumerate(combos):
        spec = dict(zip(param_names, combo))
        specs.append(spec)
        label_spec = {k: spec[k] for k in varying_keys}
        labels.append(_format_variant_label(label_spec, idx))

    if explicit_labels is not None:
        if len(explicit_labels) != len(specs):
            raise ValueError(
                f"Length of _labels ({len(explicit_labels)}) does not match number of variants ({len(specs)})"
            )
        labels = [str(x) for x in explicit_labels]

    return specs, labels


class VariantCollection(Generic[T]):
    """Functor collection of calibrated variants that forwards operations to held items.

    When instantiated via ``VariantCollection(items, ...)``, automatically dispatches
    to a specialized subclass registered for the runtime type of ``items`` (such as
    :class:`ModelVariants`, :class:`SolutionVariants`, :class:`SimulationVariants`, or
    :class:`RunResultsVariants`). Any attribute or method not explicitly overridden on
    the subclass is forwarded to each element and wrapped in a new
    :class:`VariantCollection`.
    """

    _registry: ClassVar[
        list[tuple[type | tuple[type, ...], type[VariantCollection[Any]]]]
    ] = []

    items: list[T]
    specs: list[dict[str, Any]]
    labels: list[str]

    def __init_subclass__(
        cls,
        item_type: type | tuple[type, ...] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init_subclass__(**kwargs)
        if item_type is not None:
            VariantCollection._registry.append((item_type, cls))

    def __new__(
        cls,
        items: Sequence[T],
        specs: Sequence[Mapping[str, Any]] | None = None,
        labels: Sequence[str] | None = None,
    ) -> VariantCollection[T]:
        if cls is VariantCollection and items:
            for registered_type, subcls in cls._registry:
                if all(isinstance(x, registered_type) for x in items):
                    cls = subcls  # type: ignore[assignment]
                    break
        return super().__new__(cls)

    def __init__(
        self,
        items: Sequence[T],
        specs: Sequence[Mapping[str, Any]] | None = None,
        labels: Sequence[str] | None = None,
    ) -> None:
        self.items = list(items)
        n = len(self.items)

        if specs is None:
            self.specs = [{} for _ in range(n)]
        else:
            if len(specs) != n:
                raise ValueError(
                    f"Length of specs ({len(specs)}) does not match items ({n})"
                )
            self.specs = [dict(s) for s in specs]

        if labels is None:
            self.labels = [_format_variant_label(self.specs[i], i) for i in range(n)]
        else:
            if len(labels) != n:
                raise ValueError(
                    f"Length of labels ({len(labels)}) does not match items ({n})"
                )
            self.labels = [str(lbl) for lbl in labels]

    def _wrap(self, new_items: Sequence[Any]) -> VariantCollection[Any]:
        return VariantCollection(
            items=new_items,
            specs=self.specs,
            labels=self.labels,
        )

    @property
    def parameters(self) -> list[str]:
        """List of parameter names specified in the variant calibrations."""
        seen: list[str] = []
        for s in self.specs:
            for k in s.keys():
                if k not in seen:
                    seen.append(k)
        return seen

    def __len__(self) -> int:
        return len(self.items)

    def __iter__(self) -> Iterator[T]:
        return iter(self.items)

    def __contains__(self, key: Any) -> bool:
        if isinstance(key, str) and key in self.labels:
            return True
        return key in self.items

    @overload
    def __getitem__(self, key: int) -> T: ...

    @overload
    def __getitem__(self, key: slice) -> VariantCollection[T]: ...

    @overload
    def __getitem__(self, key: str) -> T: ...

    def __getitem__(self, key: int | slice | str) -> Any:
        if isinstance(key, slice):
            return VariantCollection(
                items=self.items[key],
                specs=self.specs[key],
                labels=self.labels[key],
            )
        if isinstance(key, (int, np.integer)):
            return self.items[int(key)]
        if isinstance(key, str):
            if key in self.labels:
                idx = self.labels.index(key)
                return self.items[idx]
            raise KeyError(f"Variant label {key!r} not found in {self.labels}")
        raise TypeError(f"Invalid key type for VariantCollection: {type(key).__name__}")

    def keys(self) -> list[str]:
        return list(self.labels)

    def values(self) -> list[T]:
        return list(self.items)

    def items_with_labels(self) -> list[tuple[str, T]]:
        return list(zip(self.labels, self.items))

    def map(self, fn: Callable[[T], Any]) -> VariantCollection[Any]:
        """Apply ``fn`` to each item in the collection and wrap the results."""
        return self._wrap([fn(item) for item in self.items])

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_") or not self.items:
            raise AttributeError(
                f"{type(self).__name__!r} object has no attribute {name!r}"
            )

        attrs = [getattr(item, name) for item in self.items]
        if all(a is None for a in attrs):
            return None
        if all(callable(a) for a in attrs):

            def _mapped_call(*args: Any, **kwargs: Any) -> VariantCollection[Any]:
                return self._wrap([a(*args, **kwargs) for a in attrs])

            return _mapped_call

        return self._wrap(attrs)

    # ------------------------------------------------------------------
    # Display / Formatting
    # ------------------------------------------------------------------

    def _stage_name(self) -> str:
        if not self.items:
            return "Empty"
        return type(self.items[0]).__name__

    def __repr__(self) -> str:
        stage = self._stage_name()
        cls_name = type(self).__name__
        labels_preview = ", ".join(self.labels[:6])
        if len(self.labels) > 6:
            labels_preview += f", ... (+{len(self.labels) - 6} more)"
        return f"{cls_name}[{stage}](n={len(self)}, variants=[{labels_preview}])"

    def _repr_html_(self) -> str | None:
        stage = html.escape(self._stage_name())
        cls_name = html.escape(type(self).__name__)
        params = self.parameters
        header_cols = "".join(
            f'<th style="padding:6px 10px;border:1px solid #cbd5e1;background:#f8fafc;text-align:left;">{html.escape(p)}</th>'
            for p in params
        )
        rows: list[str] = []
        for idx, (label, spec, item) in enumerate(
            zip(self.labels, self.specs, self.items)
        ):
            param_cells = "".join(
                f'<td style="padding:6px 10px;border:1px solid #e2e8f0;">{html.escape(_format_scalar(spec.get(p, "")))}</td>'
                for p in params
            )
            rows.append(
                f"<tr>"
                f'<td style="padding:6px 10px;border:1px solid #e2e8f0;color:#64748b;">{idx}</td>'
                f'<td style="padding:6px 10px;border:1px solid #e2e8f0;font-weight:600;">{html.escape(label)}</td>'
                f"{param_cells}"
                f'<td style="padding:6px 10px;border:1px solid #e2e8f0;font-family:monospace;font-size:12px;">{html.escape(type(item).__name__)}</td>'
                f"</tr>"
            )
        return (
            f'<div style="margin:8px 0;">'
            f'<div style="font-weight:600;margin-bottom:6px;">{cls_name} ({len(self)} {stage} variants)</div>'
            f'<table style="border-collapse:collapse;font-size:13px;">'
            f"<thead><tr>"
            f'<th style="padding:6px 10px;border:1px solid #cbd5e1;background:#f8fafc;text-align:left;">#</th>'
            f'<th style="padding:6px 10px;border:1px solid #cbd5e1;background:#f8fafc;text-align:left;">Variant</th>'
            f"{header_cols}"
            f'<th style="padding:6px 10px;border:1px solid #cbd5e1;background:#f8fafc;text-align:left;">Type</th>'
            f"</tr></thead>"
            f"<tbody>{''.join(rows)}</tbody>"
            f"</table></div>"
        )


M = TypeVar("M", bound=AbstractModel)


class ModelVariants(VariantCollection[M], item_type=AbstractModel):
    """Specialized variant collection for :class:`AbstractModel` instances.

    Standard model operations (``.steady()``, ``.check()``, ``.solve()``,
    ``.perturb()``, ``.simulate()``, ``.run()``) are forwarded automatically by
    :class:`VariantCollection`. Only methods that update calibration metadata
    (``specs`` and ``labels``) are overridden here.
    """

    @property
    def models(self) -> list[M]:
        return self.items

    @property
    def is_deterministic(self) -> bool:
        return bool(self.items and all(m.is_deterministic for m in self.items))

    def recalibrate(self, **calib: Any) -> ModelVariants[M]:
        """Recalibrate all models in the collection and update variant metadata."""
        new_items = [item.recalibrate(**calib) for item in self.items]
        new_specs = [dict(s) | calib for s in self.specs]
        new_labels = [
            _format_variant_label(new_specs[i], i) for i in range(len(new_items))
        ]
        return ModelVariants(new_items, specs=new_specs, labels=new_labels)

    def variants(self, *args: Any, **kwargs: Any) -> ModelVariants[M]:
        """Expand existing model variants with an additional parameter sweep."""
        sub_specs, _ = _expand_variant_specs(args, kwargs)
        new_items: list[M] = []
        new_specs: list[dict[str, Any]] = []
        for base_item, base_spec in zip(self.items, self.specs):
            for sub_spec in sub_specs:
                merged_spec = dict(base_spec) | sub_spec
                new_items.append(base_item.recalibrate(**sub_spec))
                new_specs.append(merged_spec)
        new_labels = [
            _format_variant_label(new_specs[i], i) for i in range(len(new_items))
        ]
        return ModelVariants(new_items, specs=new_specs, labels=new_labels)


S = TypeVar("S", bound=PerturbationSolution | RecursiveDecisionRule)


class SolutionVariants(
    VariantCollection[S],
    item_type=(PerturbationSolution, RecursiveDecisionRule),
):
    """Specialized variant collection for solved decision rules.

    Forwards ``.simulate()``, ``.irfs()``, ``.moments()``, etc. automatically,
    and overrides ``.plot()`` to unify IRFs into a single comparative plot.
    """

    @property
    def solutions(self) -> list[S]:
        return self.items

    def plot(
        self,
        type: IRFType = "log-deviation",
        units: UnitsType | None = None,
        variables: list[str] | None = None,
        shocks: list[str] | str | None = None,
        T: int = 40,
        engine: str = "plotly",
        **kwargs: Any,
    ) -> Any:
        """Compute IRFs for all solution variants and unify them into one plot."""
        target_units: UnitsType = units if units is not None else type
        sim_variants = self.irfs(type=target_units, T=T)
        return sim_variants.plot(
            variables=variables,
            shocks=shocks,
            T=T,
            units=target_units,
            engine=engine,
            **kwargs,
        )


R = TypeVar("R", bound=SimulationResult)


class SimulationVariants(VariantCollection[R], item_type=SimulationResult):
    """Specialized variant collection for :class:`SimulationResult` instances.

    Provides reduction methods that unify the $M$ variant simulations into a
    single comparative plot, DataFrame, xarray DataArray, or 4D numpy tensor.
    """

    @property
    def simulations(self) -> list[R]:
        return self.items

    def in_units(self, units: UnitsType | None = None) -> np.ndarray:
        """Return a 4D ``(M, N, T+1, V)`` array across all ``M`` simulation variants."""
        arrays = [item.in_units(units) for item in self.items]
        return np.stack(arrays, axis=0)

    def to_dict(self, units: UnitsType | None = None) -> dict[str, pd.DataFrame]:
        """Return a dictionary mapping each variant label to its DataFrame."""
        return {
            label: item.to_df(units=units)
            for label, item in zip(self.labels, self.items)
        }

    def to_df(self, units: UnitsType | None = None) -> pd.DataFrame:
        """Convert simulation variants into a unified MultiIndex DataFrame indexed by ``variant``."""
        frames: dict[str, pd.DataFrame] = {}
        for label, item in zip(self.labels, self.items):
            df = item.to_df(units=units)
            if "t" in df.columns and not isinstance(df.index, pd.MultiIndex):
                df = df.set_index("t")
            elif df.index.name is None and not isinstance(df.index, pd.MultiIndex):
                df = df.copy()
                df.index.name = "t"
            frames[label] = df
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, names=["variant"])

    @property
    def df(self) -> pd.DataFrame:
        """Shorthand property for ``self.to_df()``."""
        return self.to_df()

    def to_xarray(self, units: UnitsType | None = None) -> Any:
        """Convert simulation variants to an ``xarray.DataArray`` with dimensions ``('variant', 'N', 'T', 'V')``."""
        import xarray as xr

        das = [item.to_xarray(units=units) for item in self.items]
        combined = xr.concat(das, dim="variant")
        return combined.assign_coords(variant=self.labels)

    def plot(
        self,
        variables: list[str] | None = None,
        shocks: list[str] | str | None = None,
        T: int | None = None,
        units: UnitsType | None = None,
        engine: str = "plotly",
        **kwargs: Any,
    ) -> Any:
        """Unify all simulation variants into a single comparative plot."""
        from .plots import plot_variants

        if units is None and "type" in kwargs:
            units = kwargs.pop("type")

        return plot_variants(
            self,
            variables=variables,
            shocks=shocks,
            T=T,
            units=units,
            engine=engine,
            **kwargs,
        )


RR = TypeVar("RR", bound=RunResults)


class RunResultsVariants(VariantCollection[RR], item_type=RunResults):
    """Specialized variant collection for :class:`RunResults` instances.

    Returned when running a pipeline containing ``@run: variants: ...`` or calling
    ``model.variants(...).run()``. Forwards ``.model``, ``.solution``,
    ``.simulation``, ``.residuals``, and ``.eigenvalues`` via :class:`VariantCollection`
    while unifying ``.figure``, ``.plot()``, and report rendering across all variants.
    """

    source_txt: str | None = None
    output_type: str = "html"
    mime_bundle_repr: str | None = None
    _unified_figure: Any | None = None
    _figure_computed: bool = False

    def __init__(
        self,
        items: Sequence[RR],
        specs: Sequence[Mapping[str, Any]] | None = None,
        labels: Sequence[str] | None = None,
    ) -> None:
        super().__init__(items, specs=specs, labels=labels)
        self.source_txt = self.items[0].source_txt if self.items else None
        self.output_type = self.items[0].output_type if self.items else "html"
        self.mime_bundle_repr = self.items[0].mime_bundle_repr if self.items else None
        self._unified_figure = None
        self._figure_computed = False

    @property
    def elapsed(self) -> float:
        return sum(r.elapsed or 0.0 for r in self.items)

    @property
    def errors(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for lbl, r in zip(self.labels, self.items):
            for e in r.errors:
                entry = dict(e)
                entry["message"] = f"[{lbl}] {entry.get('message', '')}"
                out.append(entry)
        return out

    @property
    def warnings(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for lbl, r in zip(self.labels, self.items):
            for w in r.warnings:
                entry = dict(w)
                entry["message"] = f"[{lbl}] {entry.get('message', '')}"
                out.append(entry)
        return out

    @property
    def _highlighting_data(self) -> list[dict[str, Any]]:
        data: list[dict[str, Any]] = []
        for r in self.items:
            data.extend(r._highlighting_data)
        return data

    @property
    def _plot_options(self) -> dict[str, Any]:
        for r in self.items:
            opts = getattr(r, "_plot_options", None)
            if isinstance(opts, dict):
                return dict(opts)
        return {}

    @property
    def _should_render_plot(self) -> bool:
        if self._figure_computed and self._unified_figure is not None:
            return True
        return any(getattr(r, "_should_render_plot", False) for r in self.items)

    @property
    def figure(self) -> Any | None:
        if self._figure_computed:
            return self._unified_figure
        if not any(r.figure is not None for r in self.items):
            return None
        sim_vc = self.simulation
        if sim_vc is None:
            return None
        plot_opts = self._plot_options
        engine = plot_opts.pop("engine", "plotly")
        self._unified_figure = sim_vc.plot(engine=engine, **plot_opts)
        self._figure_computed = True
        return self._unified_figure

    @figure.setter
    def figure(self, value: Any) -> None:
        self._unified_figure = value
        self._figure_computed = True

    def plot(self, **kwargs: Any) -> Any:
        """Plot unified simulations across all variants."""
        sim_vc = self.simulation
        if sim_vc is None:
            raise ValueError("No simulations available to plot in RunResultsVariants.")
        return sim_vc.plot(**kwargs)

    def plot_text(self, **kwargs: Any) -> str:
        """Render unified simulations across all variants as a plotext text chart."""
        sim_vc = self.simulation
        if sim_vc is None:
            return ""
        from .plots import plot_variants_plotext

        opts = dict(self._plot_options)
        if "vars" in opts and "variables" not in opts:
            opts["variables"] = opts.pop("vars")
        if "type" in opts and "units" not in opts:
            opts["units"] = opts.pop("type")
        for key in ("variables", "shocks", "T", "units"):
            if kwargs.get(key) is None and key in opts:
                kwargs[key] = opts[key]

        return plot_variants_plotext(sim_vc, **kwargs)

    def to_text(
        self,
        *,
        graphs: bool | None = None,
        color: bool | None = None,
        width: int | None = None,
        height: int | None = None,
        cols: int = 2,
        variables: list[str] | None = None,
        marker: str | None = None,
    ) -> str:
        for r in self.items:
            if r.elapsed is None:
                r.finish()

        lines: list[str] = ["RunResultsVariants", "=================="]

        base_model = next((r.model for r in self.items if r.model is not None), None)
        if base_model is not None:
            symbols = getattr(base_model, "symbols", {})
            variables_all = list(symbols.get("variables", []))
            endogenous = list(symbols.get("endogenous", []))
            exogenous = list(symbols.get("exogenous", []))
            parameters = list(symbols.get("parameters", []))
            deterministic = bool(getattr(base_model, "is_deterministic", False))

            lines.extend(
                [
                    "Model",
                    "-----",
                    f"name: {getattr(base_model, 'name', None)}",
                    f"filename: {getattr(base_model, 'filename', None)}",
                    f"variants ({len(self)}): {', '.join(self.labels)}",
                    f"deterministic: {deterministic}",
                    (
                        "symbols: "
                        f"variables={len(variables_all)}, "
                        f"endogenous={len(endogenous)}, "
                        f"exogenous={len(exogenous)}, "
                        f"parameters={len(parameters)}"
                    ),
                    f"endogenous: {RunResults._format_symbol_list(endogenous)}",
                    f"exogenous: {RunResults._format_symbol_list(exogenous)}",
                    f"parameters: {RunResults._format_symbol_list(parameters)}",
                    "",
                ]
            )
        else:
            lines.extend([f"variants ({len(self)}): {', '.join(self.labels)}", ""])

        lines.extend(["Checks", "------"])
        for lbl, r in zip(self.labels, self.items):
            lines.append(f"[{lbl}]")
            for r_line in r._residuals_summary_lines():
                lines.append(f"  {r_line}")
            for ev_line in r._eigenvalues_summary_lines():
                lines.append(f"  {ev_line}")
        lines.append("")

        lines.extend(["Outputs", "-------"])
        sol_items = [r.solution for r in self.items if r.solution is not None]
        if sol_items:
            lines.append(f"Solution: computed ({len(sol_items)}/{len(self)} variants)")
            first_sol = sol_items[0]
            dr = getattr(first_sol, "decision_rule", first_sol)
            x_shape = getattr(getattr(dr, "X", None), "shape", None)
            y_shape = getattr(getattr(dr, "Y", None), "shape", None)
            s_shape = getattr(getattr(dr, "Σ", None), "shape", None)
            if x_shape is not None or y_shape is not None or s_shape is not None:
                lines.append(
                    f"  decision rule matrices: X{x_shape}, Y{y_shape}, Σ{s_shape}"
                )
        else:
            lines.append("Solution: not computed")

        sim_results = [r for r in self.items if r.simulation is not None]
        if sim_results:
            sim_summary = sim_results[0]._simulation_summary_line().split("\n")
            sim_summary[0] = (
                f"{sim_summary[0]} ({len(sim_results)}/{len(self)} variants)"
            )
            lines.extend(sim_summary)
        else:
            lines.append("Simulation: not computed")

        if self.figure is not None:
            lines.append(f"Figure: available ({type(self.figure).__name__})")
        else:
            lines.append("Figure: not available")

        moments_items = [r.moments for r in self.items if r.moments is not None]
        if moments_items:
            shape = getattr(moments_items[0], "shape", None)
            lines.append(
                f"Moments: available{f' (shape={shape})' if shape else ''} ({len(moments_items)}/{len(self)} variants)"
            )
        else:
            lines.append("Moments: not available")
        lines.append("")

        use_graphs = self._should_render_plot if graphs is None else graphs
        if use_graphs and self.simulation is not None:
            sim_plot = self.plot_text(
                cols=cols,
                width=width,
                height=height,
                variables=variables,
                color=color,
                marker=marker,
            )
            if sim_plot.strip():
                lines.extend(["Simulation Plots", "----------------", sim_plot, ""])

        lines.extend(["Diagnostics", "-----------"])
        dummy_r = self.items[0] if self.items else RunResults()
        lines.extend(dummy_r._diagnostic_lines(title="Warnings", entries=self.warnings))
        lines.extend(dummy_r._diagnostic_lines(title="Errors", entries=self.errors))
        lines.append("")

        lines.extend(["Timing", "------", f"elapsed: {self.elapsed:.3f}s"])
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.to_text()

    @staticmethod
    def _simulation_variants_to_html(
        sim_variants: SimulationVariants[Any], **plot_opts: Any
    ) -> str:
        from .plots import _prepare_variants_nsim

        plot_vars = plot_opts.get("variables") or plot_opts.get("vars")
        plot_shocks = plot_opts.get("shocks")
        plot_T = plot_opts.get("T")
        plot_units = plot_opts.get("units") or plot_opts.get("type")
        try:
            data = _prepare_variants_nsim(
                sim_variants,
                variables=plot_vars,
                shocks=plot_shocks,
                T=plot_T,
                units=plot_units,
            ).copy()
        except Exception:
            return ""

        required = {"t", "variable", "value", "variant"}
        if not required.issubset(data.columns):
            return ""

        data["t"] = pd.to_numeric(data["t"], errors="coerce")
        data["value"] = pd.to_numeric(data["value"], errors="coerce")
        data = data[np.isfinite(data["t"]) & np.isfinite(data["value"])]
        if data.empty:
            return ""

        variables = list(dict.fromkeys(data["variable"].astype(str)))
        variants = list(sim_variants.labels)
        palette = [
            "#2563eb",
            "#dc2626",
            "#059669",
            "#d97706",
            "#7c3aed",
            "#0891b2",
            "#db2777",
            "#4f46e5",
        ]
        color_map = {v: palette[i % len(palette)] for i, v in enumerate(variants)}

        unique_shocks = (
            list(dict.fromkeys(data["shock"].astype(str)))
            if "shock" in data.columns
            else []
        )
        is_multi_shock = len(unique_shocks) > 1
        dash_styles = ["", "6 3", "2 2", "6 2 2 2"]
        shock_dash_map = {
            s: dash_styles[i % len(dash_styles)] for i, s in enumerate(unique_shocks)
        }
        extra_top = 22 if is_multi_shock else 0

        panel_width = 260
        panel_height = 170
        cols = 2
        rows = max(1, (len(variables) + cols - 1) // cols)
        svg_width = cols * panel_width
        svg_height = rows * panel_height + 28 + extra_top

        parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{svg_width}" height="{svg_height}" viewBox="0 0 {svg_width} {svg_height}" role="img" aria-label="Simulation variant charts">'
        ]
        parts.append('<rect width="100%" height="100%" fill="white"/>')

        if variants:
            legend_x = 12
            for v_lbl in variants:
                color = color_map[v_lbl]
                safe_lbl = html.escape(str(v_lbl))
                parts.append(
                    f'<line x1="{legend_x}" y1="16" x2="{legend_x + 18}" y2="16" stroke="{color}" stroke-width="2"/>'
                )
                parts.append(
                    f'<text x="{legend_x + 24}" y="20" font-size="12" fill="#334155">{safe_lbl}</text>'
                )
                legend_x += 24 + max(36, len(safe_lbl) * 7)

        if is_multi_shock:
            shock_legend_x = 12
            for s_lbl in unique_shocks:
                dash_attr = (
                    f' stroke-dasharray="{shock_dash_map[s_lbl]}"'
                    if shock_dash_map[s_lbl]
                    else ""
                )
                safe_s = html.escape(str(s_lbl))
                parts.append(
                    f'<line x1="{shock_legend_x}" y1="36" x2="{shock_legend_x + 18}" y2="36" stroke="#64748b"{dash_attr} stroke-width="2"/>'
                )
                parts.append(
                    f'<text x="{shock_legend_x + 24}" y="40" font-size="12" fill="#64748b">shock: {safe_s}</text>'
                )
                shock_legend_x += 24 + max(48, (len(safe_s) + 7) * 7)

        for index, variable in enumerate(variables):
            subset = data[data["variable"].astype(str) == variable].copy()
            if subset.empty:
                continue

            x0 = (index % cols) * panel_width
            y0 = (index // cols) * panel_height + 28 + extra_top

            left = x0 + 36
            top = y0 + 18
            plot_width = panel_width - 56
            plot_height = panel_height - 52

            xmin = float(subset["t"].min())
            xmax = float(subset["t"].max())
            ymin = float(subset["value"].min())
            ymax = float(subset["value"].max())

            if xmin == xmax:
                xmax = xmin + 1.0
            if ymin == ymax:
                pad = 1.0 if ymin == 0 else abs(ymin) * 0.1
                ymin -= pad
                ymax += pad

            def sx(value: float) -> float:
                return left + ((value - xmin) / (xmax - xmin)) * plot_width

            def sy(value: float) -> float:
                return (
                    top + plot_height - ((value - ymin) / (ymax - ymin)) * plot_height
                )

            zero_y = sy(0.0) if ymin <= 0.0 <= ymax else None
            if zero_y is not None:
                parts.append(
                    f'<line x1="{left}" y1="{zero_y:.2f}" x2="{left + plot_width}" y2="{zero_y:.2f}" stroke="#cbd5e1" stroke-width="1" stroke-dasharray="4 3"/>'
                )

            parts.append(
                f'<rect x="{left}" y="{top}" width="{plot_width}" height="{plot_height}" fill="none" stroke="#cbd5e1" stroke-width="1"/>'
            )
            parts.append(
                f'<text x="{left}" y="{y0 + 12}" font-size="13" font-weight="600" fill="#0f172a">{html.escape(str(variable))}</text>'
            )
            parts.append(
                f'<text x="{left}" y="{top + plot_height + 18}" font-size="11" fill="#64748b">{xmin:g}</text>'
            )
            parts.append(
                f'<text x="{left + plot_width - 8}" y="{top + plot_height + 18}" text-anchor="end" font-size="11" fill="#64748b">{xmax:g}</text>'
            )
            parts.append(
                f'<text x="{left - 6}" y="{top + 10}" text-anchor="end" font-size="11" fill="#64748b">{ymax:.3g}</text>'
            )
            parts.append(
                f'<text x="{left - 6}" y="{top + plot_height}" text-anchor="end" font-size="11" fill="#64748b">{ymin:.3g}</text>'
            )

            for v_lbl in variants:
                v_sub = subset[subset["variant"].astype(str) == v_lbl]
                if v_sub.empty:
                    continue
                color = color_map[v_lbl]
                group_cols = [c for c in ("shock", "n") if c in v_sub.columns]
                groups = (
                    v_sub.groupby(group_cols, sort=False)
                    if group_cols
                    else [(None, v_sub)]
                )
                for grp_key, grp in groups:
                    grp_sorted = grp.sort_values("t")
                    points = " ".join(
                        f"{sx(float(row.t)):.2f},{sy(float(row.value)):.2f}"
                        for row in grp_sorted.itertuples(index=False)
                    )
                    if points:
                        dash_val = ""
                        if is_multi_shock and isinstance(grp_key, tuple):
                            if "shock" in group_cols:
                                s_val = str(grp_key[group_cols.index("shock")])
                                dash_val = shock_dash_map.get(s_val, "")
                        elif (
                            is_multi_shock
                            and grp_key is not None
                            and "shock" in group_cols
                        ):
                            dash_val = shock_dash_map.get(str(grp_key), "")
                        dash_attr = (
                            f' stroke-dasharray="{dash_val}"' if dash_val else ""
                        )
                        parts.append(
                            f'<polyline fill="none" stroke="{color}"{dash_attr} stroke-width="2" points="{points}"/>'
                        )

        parts.append("</svg>")
        return "".join(parts)

    def _render_html_report(self) -> str:
        parts: list[str] = []
        base_model = next((r.model for r in self.items if r.model is not None), None)
        if base_model is not None and hasattr(base_model, "_repr_html_"):
            parts.append(base_model._repr_html_())

        # Clean variants summary table
        params = self.parameters
        header_cols = "".join(
            f'<th style="padding:6px 10px;border:1px solid #cbd5e1;background:#f8fafc;text-align:left;">{html.escape(p)}</th>'
            for p in params
        )
        v_rows: list[str] = []
        for idx, (label, spec) in enumerate(zip(self.labels, self.specs)):
            param_cells = "".join(
                f'<td style="padding:6px 10px;border:1px solid #e2e8f0;">{html.escape(_format_scalar(spec.get(p, "")))}</td>'
                for p in params
            )
            v_rows.append(
                f"<tr>"
                f'<td style="padding:6px 10px;border:1px solid #e2e8f0;color:#64748b;">{idx}</td>'
                f'<td style="padding:6px 10px;border:1px solid #e2e8f0;font-weight:600;">{html.escape(label)}</td>'
                f"{param_cells}"
                f"</tr>"
            )
        parts.append(
            f'<div style="margin:8px 0;">'
            f'<div style="font-weight:600;margin-bottom:6px;">Variants ({len(self)})</div>'
            f'<table style="border-collapse:collapse;font-size:13px;">'
            f"<thead><tr>"
            f'<th style="padding:6px 10px;border:1px solid #cbd5e1;background:#f8fafc;text-align:left;">#</th>'
            f'<th style="padding:6px 10px;border:1px solid #cbd5e1;background:#f8fafc;text-align:left;">Variant</th>'
            f"{header_cols}"
            f"</tr></thead>"
            f"<tbody>{''.join(v_rows)}</tbody>"
            f"</table></div>"
        )

        # Check section: Residuals and Generalized Eigenvalues across variants
        check_parts: list[str] = []
        if any(r.residuals is not None for r in self.items):
            n_eq = max(
                (
                    np.asarray(r.residuals).size
                    for r in self.items
                    if r.residuals is not None
                ),
                default=0,
            )
            eq_headers = "".join(
                f'<th style="padding:6px 10px;border:1px solid #cbd5e1;background:#f8fafc;font-size:11px;color:#64748b;">eq {i + 1}</th>'
                for i in range(n_eq)
            )
            res_rows: list[str] = []
            for lbl, r in zip(self.labels, self.items):
                if r.residuals is None:
                    cells = "".join(
                        '<td style="padding:6px 10px;border:1px solid #e2e8f0;color:#94a3b8;">—</td>'
                        for _ in range(n_eq)
                    )
                else:
                    flat = np.asarray(r.residuals, dtype=float).reshape(-1)
                    cell_list: list[str] = []
                    for val in flat:
                        is_bad = abs(float(val)) >= 1e-6
                        val_color = "#dc2626" if is_bad else "#0f172a"
                        bg_color = "background:#fef2f2;" if is_bad else ""
                        weight = "600" if is_bad else "400"
                        cell_list.append(
                            f'<td style="padding:6px 10px;border:1px solid #e2e8f0;white-space:nowrap;{bg_color}color:{val_color};font-weight:{weight};font-size:13px;">'
                            f"{html.escape(f'{float(val):.6g}')}</td>"
                        )
                    cells = "".join(cell_list)
                res_rows.append(
                    f"<tr>"
                    f'<td style="padding:6px 10px;border:1px solid #e2e8f0;font-weight:600;white-space:nowrap;">{html.escape(lbl)}</td>'
                    f"{cells}</tr>"
                )
            check_parts.append(
                '<div style="margin:10px 0 14px 0;">'
                '<div style="font-weight:600;color:#0f172a;margin-bottom:6px;">Residuals</div>'
                '<div style="overflow-x:auto;"><table style="border-collapse:collapse;">'
                f'<thead><tr><th style="padding:6px 10px;border:1px solid #cbd5e1;background:#f8fafc;font-size:11px;color:#64748b;">Variant</th>{eq_headers}</tr></thead>'
                f"<tbody>{''.join(res_rows)}</tbody></table></div></div>"
            )

        if any(r.eigenvalues is not None for r in self.items):
            n_ev = max(
                (
                    np.asarray(r.eigenvalues).size
                    for r in self.items
                    if r.eigenvalues is not None
                ),
                default=0,
            )
            ev_headers = "".join(
                f'<th style="padding:6px 10px;border:1px solid #cbd5e1;background:#f8fafc;font-size:11px;color:#64748b;">{i + 1}</th>'
                for i in range(n_ev)
            )
            ev_rows: list[str] = []
            for lbl, r in zip(self.labels, self.items):
                bk = r.bk_check
                if bk is True:
                    bk_cell = '<span style="color:#059669;font-weight:600;">Met</span>'
                elif bk is False:
                    bk_cell = (
                        '<span style="color:#dc2626;font-weight:600;">Not met</span>'
                    )
                else:
                    bk_cell = "—"
                if r.eigenvalues is None:
                    cells = "".join(
                        '<td style="padding:6px 10px;border:1px solid #e2e8f0;color:#94a3b8;">—</td>'
                        for _ in range(n_ev)
                    )
                else:
                    flat_ev = np.asarray(r.eigenvalues).reshape(-1)
                    cell_list = []
                    for val in flat_ev:
                        if np.iscomplexobj(np.asarray([val])):
                            formatted = str(val)
                        else:
                            formatted = f"{float(val):.6g}"
                        cell_list.append(
                            f'<td style="padding:6px 10px;border:1px solid #e2e8f0;white-space:nowrap;color:#0f172a;font-size:13px;">'
                            f"{html.escape(formatted)}</td>"
                        )
                    cells = "".join(cell_list)
                ev_rows.append(
                    f"<tr>"
                    f'<td style="padding:6px 10px;border:1px solid #e2e8f0;font-weight:600;white-space:nowrap;">{html.escape(lbl)}</td>'
                    f'<td style="padding:6px 10px;border:1px solid #e2e8f0;white-space:nowrap;">{bk_cell}</td>'
                    f"{cells}</tr>"
                )
            check_parts.append(
                '<div style="margin:10px 0 14px 0;">'
                '<div style="font-weight:600;color:#0f172a;margin-bottom:6px;">Generalized Eigenvalues</div>'
                '<div style="overflow-x:auto;"><table style="border-collapse:collapse;">'
                f'<thead><tr><th style="padding:6px 10px;border:1px solid #cbd5e1;background:#f8fafc;font-size:11px;color:#64748b;">Variant</th>'
                f'<th style="padding:6px 10px;border:1px solid #cbd5e1;background:#f8fafc;font-size:11px;color:#64748b;">Blanchard-Kahn</th>{ev_headers}</tr></thead>'
                f"<tbody>{''.join(ev_rows)}</tbody></table></div></div>"
            )

        if check_parts:
            parts.append("<h3>Check</h3>")
            parts.extend(check_parts)

        # Decision Rule section across variants
        sol_blocks: list[str] = []
        for idx, (lbl, r) in enumerate(zip(self.labels, self.items)):
            if r.solution is not None and hasattr(r.solution, "coefficients_as_df"):
                ss, df = r.solution.coefficients_as_df()
                open_attr = " open" if idx == 0 else ""
                sol_blocks.append(
                    f'<details{open_attr} style="border:1px solid #e2e8f0;padding:8px 12px;margin:6px 0;border-radius:4px;">'
                    f'<summary style="font-weight:600;cursor:pointer;">{html.escape(lbl)}</summary>'
                    f'<div style="margin-top:8px;">'
                    f"<h4>Steady-state</h4>{ss.to_html(index=False)}"
                    f"<h4>Jacobian</h4>{df.to_html()}"
                    f"</div></details>"
                )
        if sol_blocks:
            parts.append("<h3>Decision Rule</h3>")
            parts.append("".join(sol_blocks))

        # Moments & Simulation section
        sim_vc = self.simulation
        if sim_vc is not None:
            moment_blocks: list[str] = []
            for idx, (lbl, r) in enumerate(zip(self.labels, self.items)):
                cond_df, uncond_df = r._moments_dataframes()
                if cond_df is not None or uncond_df is not None:
                    open_attr = " open" if idx == 0 else ""
                    inner = []
                    if uncond_df is not None:
                        inner.append(
                            f"<h4>Unconditional Moments</h4>{uncond_df.to_html()}"
                        )
                    if cond_df is not None:
                        inner.append(f"<h4>Conditional Moments</h4>{cond_df.to_html()}")
                    moment_blocks.append(
                        f'<details{open_attr} style="border:1px solid #e2e8f0;padding:8px 12px;margin:6px 0;border-radius:4px;">'
                        f'<summary style="font-weight:600;cursor:pointer;">{html.escape(lbl)}</summary>'
                        f'<div style="margin-top:8px;">{"".join(inner)}</div></details>'
                    )
            if moment_blocks:
                parts.append("<h3>Moments</h3>")
                parts.append("".join(moment_blocks))

            if self._should_render_plot:
                sim_html = self._simulation_variants_to_html(
                    sim_vc, **self._plot_options
                )
                if sim_html:
                    parts.append("<h3>Simulation</h3>")
                    parts.append(sim_html)
                elif self.figure is not None:
                    parts.append("<h3>Simulation</h3>")
                    parts.append(RunResults._figure_to_html(self.figure))
        elif self._should_render_plot and self.figure is not None:
            parts.append("<h3>Simulation</h3>")
            parts.append(RunResults._figure_to_html(self.figure))

        for e in self.errors:
            parts.append(f"<pre style='color:red'>{html.escape(e['message'])}</pre>")
        return "<br>".join(parts)

    def _repr_html_(self) -> str | None:
        if self.output_type != "html":
            return None
        return self._render_html_report()

    def _repr_markdown_(self) -> str | None:
        if str(self.output_type).lower() == "text" and self.mime_bundle_repr is None:
            return None

        from math import nan
        from dyno.errors import ParserError

        blocks: list[str] = []

        # 1. Parser errors (if any)
        exception_errors = [
            e for e in self.errors if isinstance(e.get("_exception"), Exception)
        ]
        parser_errors = [
            e["_exception"]
            for e in exception_errors
            if isinstance(e.get("_exception"), ParserError)
        ]
        if parser_errors:
            for e in parser_errors:
                has_details = hasattr(e, "details") and e.details is not None
                err_lines = [f":::{{error}} {str(e)}"]
                if has_details:
                    err_lines.extend([":class: dropdown", "```", str(e), "```"])
                err_lines.append(":::")
                blocks.append("\n".join(err_lines))
            blocks.append("---")

        # 2. Model header, Calibration dropdown, Equations dropdown
        base_model = next((r.model for r in self.items if r.model is not None), None)
        if base_model is not None:
            fmt_list = lambda seq: ", ".join(f"`{x}`" for x in seq)
            variants_list = fmt_list(self.labels)
            vars_all = base_model.symbols.get("variables", [])
            vars_exo = base_model.symbols.get("exogenous", [])
            vars_endo = base_model.symbols.get("endogenous", [])
            params_all = base_model.symbols.get("parameters", [])

            n_equations = len(getattr(base_model, "equations", []))
            header_lines = [
                f"# Report: {base_model.name}",
                "",
                f"- *filename*:  {base_model.filename}",
                f"- *name*:  {base_model.name}",
                f"- *variants* ({len(self)}):  {variants_list}",
                f"- *variables* ({len(vars_all)}):      {fmt_list(vars_all)}",
                f"    - *exogenous* ({len(vars_exo)}):  {fmt_list(vars_exo)}",
                f"    -  *endogenous* (**{len(vars_endo)}**):  {fmt_list(vars_endo)}",
                f"- *equations*({n_equations})",
                f"- *{len(params_all)} parameters*:    {fmt_list(params_all)}",
            ]
            blocks.append("\n".join(header_lines))

            # Calibration dropdown: comparative tables across variants
            param_cols: dict[str, list[Any]] = {}
            steady_cols: dict[str, list[Any]] = {}
            for lbl, r in zip(self.labels, self.items):
                m = r.model or base_model
                c_dict = m.context.get("constants", {})
                s_dict = m.context.get("steady_states", {})
                param_cols[lbl] = [c_dict.get(p, nan) for p in params_all]
                steady_cols[lbl] = [s_dict.get(v, nan) for v in vars_all]

            params_df = pd.DataFrame(param_cols, index=params_all)
            steady_df = pd.DataFrame(steady_cols, index=vars_all)

            calib_block = "\n".join(
                [
                    ":::{dropdown} Calibration",
                    "Parameter values",
                    RunResults._to_html_table(params_df),
                    "Steady state values",
                    RunResults._to_html_table(steady_df),
                    ":::",
                ]
            )
            blocks.append(calib_block)

            # Equations dropdown
            eq_content = ""
            if hasattr(base_model, "symbolic") and hasattr(
                base_model.symbolic, "equations_table_markdown"
            ):
                eq_content = base_model.symbolic.equations_table_markdown()
            elif hasattr(base_model, "latex_equations"):
                eq_content = base_model.latex_equations()

            blocks.append(f":::{{dropdown}} Equations\n{eq_content}\n:::")
            blocks.append("---")

        # 3. Check section (Residuals and Blanchard-Kahn / Eigenvalues)
        has_residuals = any(r.residuals is not None for r in self.items)
        has_eigenvalues = any(r.eigenvalues is not None for r in self.items)
        if has_residuals or has_eigenvalues:
            check_lines = ["## Check", ""]

            if has_residuals:
                res_ok = all(
                    bool(np.max(np.abs(r.residuals)) < 1e-6)
                    for r in self.items
                    if r.residuals is not None
                )
                admonition = (
                    ":::{tip} Residuals are zero"
                    if res_ok
                    else ":::{warning} Residuals are not zero"
                )
                res_cols: dict[str, Any] = {}
                n_eq = 0
                for lbl, r in zip(self.labels, self.items):
                    if r.residuals is not None:
                        arr = np.asarray(r.residuals, dtype=float).reshape(-1)
                        n_eq = max(n_eq, len(arr))
                        res_cols[lbl] = arr
                eq_index = [f"eq {i + 1}" for i in range(n_eq)]
                res_df = pd.DataFrame(res_cols, index=eq_index)
                check_lines.extend(
                    [
                        admonition,
                        ":class: dropdown",
                        RunResults._to_html_table(res_df),
                        ":::",
                        "",
                    ]
                )

            if has_eigenvalues:
                bk_all = all(
                    r.bk_check is True for r in self.items if r.eigenvalues is not None
                )
                admonition = (
                    ":::{tip} Blanchard-Kahn conditions are met"
                    if bk_all
                    else ":::{warning} Blanchard-Kahn conditions are not met"
                )
                ev_cols: dict[str, Any] = {}
                for lbl, r in zip(self.labels, self.items):
                    if r.eigenvalues is not None:
                        ev_cols[lbl] = np.asarray(r.eigenvalues).reshape(-1)
                ev_df = pd.DataFrame(ev_cols)
                check_lines.extend(
                    [
                        admonition,
                        ":class: dropdown",
                        "Sorted by modulus:",
                        RunResults._to_html_table(ev_df),
                        ":::",
                        "",
                    ]
                )

            check_lines.append("---")
            blocks.append("\n".join(check_lines))

        # 4. Solution section (Recursive Decision Rule tabbed by variant)
        sol_entries = [
            (lbl, r.solution.coefficients_as_df())
            for lbl, r in zip(self.labels, self.items)
            if r.solution is not None and hasattr(r.solution, "coefficients_as_df")
        ]
        if sol_entries:
            sol_lines = [
                "## Solution",
                "",
                ":::::{dropdown} Recursive Decision Rule",
                "",
                r"$$y_t = \overline{y} + A (y_{t-1} - \overline{y}) + B \varepsilon_t$$",
                r"$$\epsilon_t \sim \mathcal{N}(0, \Sigma)$$",
                "",
                "::::{tab-set}",
            ]
            for lbl, jacs in sol_entries:
                sol_lines.extend(
                    [
                        f":::{{tab-item}} {lbl}",
                        ":sync: variant",
                        "",
                        "### Steady-state",
                        "",
                        RunResults._to_html_table(jacs[0]),
                        "",
                        "### Jacobian",
                        "",
                        RunResults._to_html_table(jacs[1]),
                        "",
                        ":::",
                    ]
                )
            sol_lines.extend(["::::", "", ":::::", "", "---"])
            blocks.append("\n".join(sol_lines))

        # 5. Simulation section (Moments + IRFs tabbed by variant + optional plot)
        if any(r.simulation is not None for r in self.items):
            sim_lines = ["## Simulation", ""]

            cond_entries: list[tuple[str, pd.DataFrame]] = []
            uncond_entries: list[tuple[str, pd.DataFrame]] = []
            for lbl, r in zip(self.labels, self.items):
                c_df, u_df = r._moments_dataframes()
                if c_df is not None:
                    cond_entries.append((lbl, c_df))
                if u_df is not None:
                    uncond_entries.append((lbl, u_df))

            if uncond_entries:
                sim_lines.extend(
                    [
                        ":::::{dropdown} Unconditional Moments",
                        "::::{tab-set}",
                    ]
                )
                for lbl, u_df in uncond_entries:
                    sim_lines.extend(
                        [
                            f":::{{tab-item}} {lbl}",
                            ":sync: variant",
                            RunResults._to_html_table(u_df),
                            ":::",
                        ]
                    )
                sim_lines.extend(["::::", ":::::", ""])

            if cond_entries:
                sim_lines.extend(
                    [
                        ":::::{dropdown} Conditional Moments",
                        "::::{tab-set}",
                    ]
                )
                for lbl, c_df in cond_entries:
                    sim_lines.extend(
                        [
                            f":::{{tab-item}} {lbl}",
                            ":sync: variant",
                            RunResults._to_html_table(c_df),
                            ":::",
                        ]
                    )
                sim_lines.extend(["::::", ":::::", ""])

            # IRFs / Simulation trajectories dropdown
            sim_lines.extend(
                [
                    ":::::{dropdown} IRFS",
                    "",
                    "::::{tab-set}",
                ]
            )
            for lbl, r in zip(self.labels, self.items):
                sim = r.simulation
                if sim is None:
                    continue
                sim_lines.extend(
                    [
                        f":::{{tab-item}} {lbl}",
                        ":sync: variant",
                    ]
                )
                if isinstance(sim, dict):
                    keys = list(sim.keys())
                    if len(keys) == 1:
                        sim_lines.append(RunResults._to_html_table(sim[keys[0]]))
                    else:
                        for k in keys:
                            sim_lines.append(f"#### `{k}`")
                            sim_lines.append(RunResults._to_html_table(sim[k]))
                else:
                    to_df_fn = getattr(sim, "to_df", None)
                    if callable(to_df_fn):
                        sim_lines.append(RunResults._to_html_table(to_df_fn()))
                    else:
                        sim_lines.append(RunResults._to_html_table(sim))
                sim_lines.append(":::")
            sim_lines.extend(["::::", "", "::::::"])

            sim_vc = self.simulation
            if self._should_render_plot and sim_vc is not None:
                sim_svg = self._simulation_variants_to_html(
                    sim_vc, **self._plot_options
                )
                if sim_svg:
                    sim_lines.extend(["", sim_svg])

            if base_model is not None and base_model.checks.get("deterministic", False):
                sim_lines.append("---")

            blocks.append("\n".join(sim_lines))

        return "\n\n".join(blocks)

    def _repr_mimebundle_(
        self,
        include: list[str] | None = None,
        exclude: list[str] | None = None,
    ) -> dict[str, Any]:
        data: dict[str, Any] = {}
        highlighting = self._highlighting_data
        if highlighting:
            data["application/vnd.jupyterlab-dyno.highlighting+json"] = highlighting

        mode = self.mime_bundle_repr
        if mode not in {None, "markdown", "html", "text"}:
            mode = None

        output_mode = str(self.output_type).lower()
        default_markdown = output_mode != "text"
        default_html = output_mode == "html"

        if mode == "markdown" or (mode is None and default_markdown):
            md = self._repr_markdown_()
            if md:
                data["text/markdown"] = md
        if mode == "html" or (mode is None and default_html):
            data["text/html"] = self._render_html_report()
        if mode in {None, "text"}:
            data["text/plain"] = repr(self)

        if include is not None:
            include_set = set(include)
            data = {k: v for k, v in data.items() if k in include_set}
        if exclude is not None:
            exclude_set = set(exclude)
            data = {k: v for k, v in data.items() if k not in exclude_set}
        return data

    def display(self) -> None:
        try:
            from IPython.display import HTML, Markdown, display
        except ImportError:
            self.console_display()
            return

        output_mode = str(self.output_type).lower()
        if output_mode == "html":
            display(HTML(self._render_html_report()))
        elif output_mode == "text":
            display({"text/plain": repr(self)}, raw=True)
        else:
            md = self._repr_markdown_()
            if md:
                display(Markdown(md))
            if self.figure is not None:
                display(self.figure)

    def console_display(self) -> None:
        print(self.to_text(graphs=True, color=True))
