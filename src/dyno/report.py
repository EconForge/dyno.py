from __future__ import annotations

import html
import time
import os
import numpy as np
import tempita

from dyno.errors import ParserError

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from dyno.model import AbstractModel
    from dyno.solver import PerturbationSolution
    import pandas as pd


# ---------------------------------------------------------------------------
# Markdown template (used by RunResults._repr_markdown_)
# ---------------------------------------------------------------------------

template = tempita.Template(
    r"""
{[default model=None]}
{[default dr=None]}
{[default bk_check=None]}
{[default sim=None]}
{[default fig=None]}
{[default eigenvalues=None]}
                            

{[if len(parser_errors)>0]}      

{[for e in parser_errors]}
                                     
:::{error} {[str(e)]}
{[if hasattr(e,'details') and e.details is not None]}
:class: dropdown
```
{[str(e)]}
```
{[endif]}
:::
                            
{[endfor]} 

---

{[endif]}


{[if model is not None]}

# Report: {[model.name]}
         

- *filename*:  {[model.filename]}
- *name*:  {[model.name]}
- *variables* ({[ len(model.symbols['variables']) ]}):      {[str.join(", ", map('`{}`'.format,model.symbols['variables']))]}
    - *exogenous* ({[ len(model.symbols['exogenous']) ]}):  {[str.join(", ", map('`{}`'.format,model.symbols['exogenous']))]}
    -  *endogenous* (**{[ len(model.symbols['endogenous']) ]}**):  {[str.join(", ", map('`{}`'.format,model.symbols['endogenous']))]}
- *equations*({[ len(model.equations) ]})
- *{[ len(model.symbols['parameters']) ]} parameters*:    {[str.join(", ", map('`{}`'.format,model.symbols['parameters']))]}

:::{dropdown} Calibration
Parameter values
```{code} python
{[ str(model.context['constants']) ]}
```
Steady state values
```{code} python
{[ steady_values ]}
```
:::

:::{dropdown} Equations                    
{[if hasattr(model,'latex_equations')]}
{[ model.latex_equations() ]}
{[endif]}
:::     

---
{[endif]}


{[if eigenvalues is not None and dr is None]}

## Eigenvalue Analysis

{[py: import numpy as _np; _evs_mod = _np.abs(eigenvalues); _n = len(_evs_mod)//2; _bk = bool(_evs_mod[_n-1] < 1 < _evs_mod[_n]) if _n > 0 else None]}
{[if _bk]}
:::{tip} Blanchard-Kahn conditions are met
{[else]}
:::{warning} Blanchard-Kahn conditions are not met
{[endif]}
:class: dropdown
Generalized Eigenvalues (sorted by modulus):
```{code} python
{[eigenvalues]}
```
:::

---
{[endif]}


{[if dr is not None]}
                    
## Solution

                                                   
{[if residuals is not None and not abs(residuals).max() < 1e-6]}
:::{warning} Non zero residuals
:class: dropdown
The model has non zero residuals after calibration. This may be due to a missing steady-state
calculation or an error in the model equations.
```{code} python
{[ str(residuals) ]}
```
:::
{[endif]}

                                       

{[if bk_check]}
:::{tip} Blanchard Kan conditions are met
{[else]}
:::{warning} Blanchard Kan conditions are not met
{[endif]}
:class: dropdown
System Eigenvalues:
```{code} python
{[dr.evs.real]}
```
:::

                                                            


:::{dropdown} Recursive Decision Rule
                            
$$y_t = \overline{y} + A (y_{t-1} - \overline{y}) + B \varepsilon_t$$
$$\epsilon_t \sim \mathcal{N}(0, \Sigma)$$

### Steady-state

{[to_html_table(jacs[0])]}
                            
### Jacobian 

{[to_html_table(jacs[1])]}

:::
---
{[endif]}
                            

{[if sim is not None and model.checks['deterministic']==False]}

## Simulation

:::::{dropdown} IRFS
                            
::::{tab-set}     
{[for k in sim.keys()]}
:::{tab-item} {[k]}
:sync: tab1
{[to_html_table(sim[k])]}
:::
{[endfor]}
::::

::::::
              
{[endif]}
                            

{[if sim is not None and model.checks['deterministic']==True]}

## Simulation

:::::{dropdown} IRFS
                            
::::{tab-set}     
{[for k in sim.keys()]}
:::{tab-item} {[k]}
:sync: tab1
{[to_html_table(sim[k])]}
:::
{[endfor]}
::::

::::::
---                   
{[endif]}
                            

# {[if len(unhandled_errors)>0]}                     
# {[for er in unhandled_errors]}
# ```
# {[str(er)]}
# ```
# {[endfor]}
# {[endif]}
""",
    delimiters=("{[", "]}"),
)


# ---------------------------------------------------------------------------
# RunResults — unified result container for all model execution paths
# ---------------------------------------------------------------------------


class RunResults:
    """Unified result container returned by model ``run()`` methods and ``dsge_report``.

    All fields are optional and populated progressively during the pipeline.
    Display methods adapt to whatever data is available.
    """

    def __init__(
        self,
        model: AbstractModel | None = None,
        *,
        source_txt: str | None = None,
    ) -> None:
        self.model: AbstractModel | None = model
        self.source_txt: str | None = source_txt

        # Pipeline outputs
        self.residuals: np.ndarray | None = None
        self.solution: PerturbationSolution | None = None
        self.simulation: dict | pd.DataFrame | None = None
        self.figure: Any | None = None
        self.eigenvalues: np.ndarray | None = None

        # Structured diagnostics: list of {line, type, message}
        self.errors: list[dict[str, Any]] = []
        self.warnings: list[dict[str, Any]] = []

        # Timing
        self._t_start: float = time.time()
        self.elapsed: float | None = None

    # -- Diagnostic helpers --------------------------------------------------

    def add_error(self, message: str, *, line: int | None = None) -> None:
        entry: dict[str, Any] = {"type": "error", "message": message}
        if line is not None:
            entry["line"] = line
        self.errors.append(entry)

    def add_warning(self, message: str, *, line: int | None = None) -> None:
        entry: dict[str, Any] = {"type": "warning", "message": message}
        if line is not None:
            entry["line"] = line
        self.warnings.append(entry)

    def finish(self) -> None:
        self.elapsed = time.time() - self._t_start

    # -- Blanchard-Kahn check ------------------------------------------------

    @property
    def bk_check(self) -> bool | None:
        if self.solution is None or self.solution.evs is None:
            return None
        evs = abs(self.solution.evs)
        n = len(evs) // 2
        if n == 0:
            return None
        return bool(evs[n - 1] < 1 < evs[n])

    # -- Display: Markdown ---------------------------------------------------

    @staticmethod
    def _to_html_table(value: Any) -> str:
        if hasattr(value, "to_html"):
            return value.to_html()
        if hasattr(value, "to_frame"):
            return value.to_frame().to_html()
        return f"<pre>{value}</pre>"

    @staticmethod
    def _figure_to_html(figure: Any) -> str:
        if hasattr(figure, "to_html"):
            try:
                return figure.to_html(full_html=False, include_plotlyjs="cdn")
            except TypeError:
                try:
                    return figure.to_html()
                except TypeError:
                    pass
        if hasattr(figure, "_repr_html_"):
            return figure._repr_html_()
        return f"<pre>{figure}</pre>"

    @staticmethod
    def _vector_to_horizontal_html(
        values: Any,
        *,
        title: str,
        value_formatter: str = "{:.6g}",
    ) -> str:
        arr = np.asarray(values)
        if arr.size == 0:
            return ""

        flat = arr.reshape(-1)
        cells = []
        for index, value in enumerate(flat, start=1):
            if np.iscomplexobj(np.asarray([value])):
                formatted = str(value)
            else:
                formatted = value_formatter.format(float(value))
            cells.append(
                "<td style=\"padding:6px 10px;border:1px solid #e2e8f0;white-space:nowrap;\">"
                f"<div style=\"font-size:11px;color:#64748b;\">{index}</div>"
                f"<div style=\"font-size:13px;color:#0f172a;\">{html.escape(formatted)}</div>"
                "</td>"
            )

        return (
            f"<div style=\"margin:10px 0 14px 0;\"><div style=\"font-weight:600;color:#0f172a;margin-bottom:6px;\">{html.escape(title)}</div>"
            "<div style=\"overflow-x:auto;\"><table style=\"border-collapse:collapse;\"><tr>"
            + "".join(cells)
            + "</tr></table></div></div>"
        )

    @staticmethod
    def _simulation_to_html(simulation: Any) -> str:
        import pandas as pd

        from dyno.simul import sim_to_nsim

        if isinstance(simulation, dict):
            data = sim_to_nsim(simulation).copy()
        elif hasattr(simulation, "melt"):
            data = simulation.copy()
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

        variables = list(dict.fromkeys(data["variable"].astype(str)))
        shocks = list(dict.fromkeys(data["shock"].astype(str)))
        colors = [
            "#0f766e",
            "#dc2626",
            "#2563eb",
            "#ca8a04",
            "#7c3aed",
            "#ea580c",
        ]
        color_map = {shock: colors[i % len(colors)] for i, shock in enumerate(shocks)}

        panel_width = 260
        panel_height = 170
        cols = 2
        rows = max(1, (len(variables) + cols - 1) // cols)
        svg_width = cols * panel_width
        svg_height = rows * panel_height + 28

        parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{svg_width}" height="{svg_height}" viewBox="0 0 {svg_width} {svg_height}" role="img" aria-label="Simulation charts">'
        ]
        parts.append('<rect width="100%" height="100%" fill="white"/>')

        if shocks:
            legend_x = 12
            for shock in shocks:
                color = color_map[shock]
                safe_shock = html.escape(str(shock))
                parts.append(f'<line x1="{legend_x}" y1="16" x2="{legend_x + 18}" y2="16" stroke="{color}" stroke-width="2"/>')
                parts.append(f'<text x="{legend_x + 24}" y="20" font-size="12" fill="#334155">{safe_shock}</text>')
                legend_x += 24 + max(36, len(safe_shock) * 7)

        for index, variable in enumerate(variables):
            subset = data[data["variable"].astype(str) == variable].copy()
            subset = subset.sort_values(["shock", "t"])
            if subset.empty:
                continue

            x0 = (index % cols) * panel_width
            y0 = (index // cols) * panel_height + 28

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
                return top + plot_height - ((value - ymin) / (ymax - ymin)) * plot_height

            zero_y = sy(0.0) if ymin <= 0.0 <= ymax else None
            if zero_y is not None:
                parts.append(
                    f'<line x1="{left}" y1="{zero_y:.2f}" x2="{left + plot_width}" y2="{zero_y:.2f}" stroke="#cbd5e1" stroke-width="1" stroke-dasharray="4 3"/>'
                )

            parts.append(f'<rect x="{left}" y="{top}" width="{plot_width}" height="{plot_height}" fill="none" stroke="#cbd5e1" stroke-width="1"/>')
            parts.append(f'<text x="{left}" y="{y0 + 12}" font-size="13" font-weight="600" fill="#0f172a">{html.escape(str(variable))}</text>')
            parts.append(f'<text x="{left}" y="{top + plot_height + 18}" font-size="11" fill="#64748b">{xmin:g}</text>')
            parts.append(f'<text x="{left + plot_width - 8}" y="{top + plot_height + 18}" text-anchor="end" font-size="11" fill="#64748b">{xmax:g}</text>')
            parts.append(f'<text x="{left - 6}" y="{top + 10}" text-anchor="end" font-size="11" fill="#64748b">{ymax:.3g}</text>')
            parts.append(f'<text x="{left - 6}" y="{top + plot_height}" text-anchor="end" font-size="11" fill="#64748b">{ymin:.3g}</text>')

            for shock in shocks:
                shock_subset = subset[subset["shock"].astype(str) == shock]
                if shock_subset.empty:
                    continue
                points = " ".join(
                    f"{sx(float(row.t)):.2f},{sy(float(row.value)):.2f}"
                    for row in shock_subset.itertuples(index=False)
                )
                if points:
                    color = color_map[shock]
                    parts.append(f'<polyline fill="none" stroke="{color}" stroke-width="2" points="{points}"/>')

        parts.append("</svg>")
        return "".join(parts)

    def _repr_markdown_(self) -> str:
        if self.elapsed is None:
            self.finish()

        import traceback as tb_mod
        import altair
        from math import nan

        # Separate parser errors from other collected exceptions
        exception_errors = [
            e for e in self.errors if isinstance(e.get("_exception"), Exception)
        ]
        parser_errors = [
            e["_exception"]
            for e in exception_errors
            if isinstance(e.get("_exception"), ParserError)
        ]
        unhandled_errors = [
            e["_exception"]
            for e in exception_errors
            if not isinstance(e.get("_exception"), ParserError)
        ]
        error_lines = [str(e.line) for e in parser_errors if hasattr(e, "line")]

        context: dict[str, Any] = {
            "traceback": tb_mod,
            "errors": [
                e.get("_exception") for e in exception_errors if "_exception" in e
            ],
            "parser_errors": parser_errors,
            "unhandled_errors": unhandled_errors,
            "error_lines": error_lines,
            "alt": altair,
            "to_html_table": self._to_html_table,
        }

        model = self.model
        if model is not None:
            steady_values = str(
                {
                    v: model.context["steady_states"].get(v, nan)
                    for v in model.symbols["variables"]
                }
            )
            context["steady_values"] = steady_values

        dr = self.solution
        if dr is not None:
            bk = self.bk_check
            context["bk_check"] = bk
            context["jacs"] = dr.coefficients_as_df()

        d: dict[str, Any] = {
            "model": model,
            "residuals": self.residuals,
            "dr": dr,
            "sim": self.simulation,
            "fig": self.figure,
            "eigenvalues": self.eigenvalues,
        }
        d.update(context)

        txt = template.substitute(**d)

        for e in unhandled_errors:
            print("Unhandled error:")
            print(e)

        return txt

    # -- Display: HTML (rich console) ----------------------------------------

    def _repr_html_(self) -> str:
        parts: list[str] = []
        if self.model is not None and hasattr(self.model, "_repr_html_"):
            parts.append(self.model._repr_html_())

        check_parts: list[str] = []
        if self.residuals is not None:
            check_parts.append(
                self._vector_to_horizontal_html(self.residuals, title="Residuals")
            )
        if self.eigenvalues is not None:
            check_parts.append(
                self._vector_to_horizontal_html(
                    self.eigenvalues,
                    title="Generalized Eigenvalues",
                )
            )
        if check_parts:
            parts.append("<h3>Check</h3>")
            parts.extend(check_parts)

        if self.solution is not None and hasattr(self.solution, "_repr_html_"):
            parts.append(self.solution._repr_html_())
        if self.simulation is not None:
            sim_html = self._simulation_to_html(self.simulation)
            if sim_html:
                parts.append("<h3>Simulation</h3>")
                parts.append(sim_html)
        elif self.figure is not None:
            parts.append("<h3>Simulation</h3>")
            parts.append(self._figure_to_html(self.figure))
        for e in self.errors:
            parts.append(f"<pre style='color:red'>{e['message']}</pre>")
        return "<br>".join(parts)

    # -- Display: JupyterLab highlighting ------------------------------------

    @property
    def _highlighting_data(self) -> list[dict[str, Any]]:
        data: list[dict[str, Any]] = []
        for entry in self.errors + self.warnings:
            if "line" in entry:
                data.append(
                    {
                        "line": entry["line"],
                        "type": entry["type"],
                        "message": entry["message"],
                    }
                )
        return data

    def jupyter_display(self) -> None:
        try:
            from IPython.display import display, Markdown
        except ImportError:
            self.console_display()
            return

        if self.elapsed is None:
            self.finish()

        highlighting = self._highlighting_data
        if highlighting:
            display(
                {"application/vnd.jupyterlab-dyno.highlighting+json": highlighting},
                raw=True,
            )

        display(Markdown(self._repr_markdown_()))

        if self.figure is not None:
            display(self.figure)

        display(Markdown("---"))

    def console_display(self) -> None:
        if self.elapsed is None:
            self.finish()

        if self.model is not None:
            print(repr(self.model))

        if self.residuals is not None:
            r = self.residuals
            if abs(r).max() < 1e-6:
                print("Residuals: OK")
            else:
                print(f"Residuals: max |r| = {abs(r).max():.2e}")

        if self.solution is not None:
            bk = self.bk_check
            if bk is True:
                print("Blanchard-Kahn conditions: met")
            elif bk is False:
                print("Blanchard-Kahn conditions: NOT met")
            print(f"Solution: computed")

        if self.simulation is not None:
            if isinstance(self.simulation, dict):
                print(f"IRFs: {len(self.simulation)} shock(s)")
            else:
                print(f"Simulation: computed")

        for e in self.errors:
            print(f"ERROR: {e['message']}")

        print(f"Elapsed: {self.elapsed:.3f}s")

    # -- Plain text ----------------------------------------------------------

    def __repr__(self) -> str:
        parts = []
        if self.model is not None:
            parts.append(f"model={self.model.name!r}")
        if self.solution is not None:
            parts.append("solution=<computed>")
        if self.simulation is not None:
            parts.append("simulation=<computed>")
        if self.errors:
            parts.append(f"errors={len(self.errors)}")
        if self.elapsed is not None:
            parts.append(f"elapsed={self.elapsed:.3f}s")
        return f"RunResults({', '.join(parts)})"


# ---------------------------------------------------------------------------
# Backward-compatible aliases
# ---------------------------------------------------------------------------

# Keep the old names importable; they now point to RunResults.
Report = RunResults
DynareRunResults = RunResults
DynoRunResults = RunResults


# ---------------------------------------------------------------------------
# dsge_report — JupyterLab entry point
# ---------------------------------------------------------------------------


def _create_model(
    txt: str | None, filename: str | os.PathLike[str] | None, **options
) -> "AbstractModel":
    if filename is not None:
        filename = os.fspath(filename)

    if txt is not None:
        if filename is None:
            filename = "unknown"
    elif filename is not None:
        with open(filename, encoding="utf-8") as f:
            txt = f.read()
    else:
        raise ValueError("Either `txt` or `filename` must be provided.")

    if filename.endswith(".mod"):
        preprocessor = options.get("modfile-preprocessor", "dynare")
        if preprocessor == "dynare":
            from dyno.dynare_model import DynareModel

            return DynareModel(filename=filename, txt=txt)
        else:
            from dyno.dyno_model import DynoModel

            return DynoModel(filename=filename, txt=txt)
    elif filename.endswith((".yaml", ".yml")):
        from dyno.dyno_model import DynoModel

        return DynoModel(filename=filename, txt=txt)
    elif filename.endswith(".dyno"):
        from dyno.dyno_model import DynoModel

        return DynoModel(filename=filename, txt=txt)
    else:
        raise ValueError("Unsupported Model type")


def dsge_report(
    txt: str | None = None,
    filename: str | os.PathLike[str] | None = None,
    **options,
) -> RunResults:

    check_output = options.get("check_output", False)
    results = RunResults(source_txt=txt)

    if check_output:
        d: dict[str, Any] = {}
        try:
            exec(txt, d, d)  # noqa: S102  — preserved existing behaviour
        except Exception as e:
            results.add_error(str(e))
            return results
        try:
            return d["html"]
        except Exception as e:
            results.add_error(str(e))
            return results

    try:
        model = _create_model(txt, filename, **options)
        results = model.run(default_pipeline=True)
        results.source_txt = txt

    except Exception as e:
        results.add_error(str(e), line=getattr(e, "line", None))
        results.errors[-1]["_exception"] = e
        results.finish()
        results.jupyter_display()  # show partial results + errors
        return results

    results.jupyter_display()
    return results
