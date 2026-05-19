from __future__ import annotations

import html
import time
import os
import numpy as np
import tempita

from dyno.errors import ParserError, SteadyStateError

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
{[default residuals=None]}
{[default moments=None]}
                            

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
{[if hasattr(model, 'symbolic') and hasattr(model.symbolic, 'equations_table_markdown')]}
{[ model.symbolic.equations_table_markdown() ]}
{[elif hasattr(model,'latex_equations')]}
{[ model.latex_equations() ]}
{[endif]}
:::     

---
{[endif]}

{[if residuals is not None or eigenvalues is not None]}

## Check

{[if residuals is not None]}
{[py: import numpy as _np; _res_ok = bool(_np.max(_np.abs(residuals)) < 1e-6)]}
{[if _res_ok]}
:::{tip} Residuals are zero
{[else]}
:::{warning} Residuals are not zero
{[endif]}
:class: dropdown
```{code} python
{[ str(residuals) ]}
```
:::
{[endif]}

{[if eigenvalues is not None]}
{[py: import numpy as _np; _evs_mod = _np.abs(eigenvalues); _n = len(_evs_mod)//2; _bk = bool(_evs_mod[_n-1] < 1 < _evs_mod[_n]) if _n > 0 else None]}
{[if _bk]}
:::{tip} Blanchard-Kahn conditions are met
{[else]}
:::{warning} Blanchard-Kahn conditions are not met
{[endif]}
:class: dropdown
Sorted by modulus:
```{code} python
{[eigenvalues]}
```
:::
{[endif]}

---

{[endif]}




{[if dr is not None]}
                    
## Solution
                         

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

{[if moments is not None]}
:::{dropdown} Unconditional Moments
{[to_html_table(moments_df)]}
:::
{[endif]}

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

{[if moments is not None]}
:::{dropdown} Unconditional Moments
{[to_html_table(moments_df)]}
:::
{[endif]}

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
        output_type: str = "markdown",
        mime_bundle_repr: str | None = None,
    ) -> None:
        self.model: AbstractModel | None = model
        self.source_txt: str | None = source_txt
        # Keep backward-compatible rendering policy: HTML is opt-in.
        self.output_type: str = output_type
        # MIME bundle policy: None => include all rich reprs; else one of
        # {"markdown", "html", "text"} to include only that representation.
        self.mime_bundle_repr: str | None = mime_bundle_repr

        # Pipeline outputs
        self.residuals: np.ndarray | None = None
        self.solution: PerturbationSolution | None = None
        self.simulation: dict | pd.DataFrame | None = None
        self.figure: Any | None = None
        self.eigenvalues: np.ndarray | None = None
        self.moments: np.ndarray | None = None

        # Structured diagnostics: list of {line, type, message}
        self.errors: list[dict[str, Any]] = []
        self.warnings: list[dict[str, Any]] = []

        # Timing
        self._t_start: float = time.time()
        self.elapsed: float | None = None

    # -- Diagnostic helpers --------------------------------------------------

    def add_error(
        self,
        message: str,
        *,
        line: int | None = None,
        column: int | None = None,
    ) -> None:
        entry: dict[str, Any] = {"type": "error", "message": message}
        if line is not None:
            entry["line"] = line
        if column is not None:
            entry["column"] = column
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
        tol: float | None = None,
        labels: list[str] | None = None,
    ) -> str:
        arr = np.asarray(values)
        if arr.size == 0:
            return ""

        flat = arr.reshape(-1)
        cells = []
        for index, value in enumerate(flat, start=1):
            if np.iscomplexobj(np.asarray([value])):
                formatted = str(value)
                is_bad = False
            else:
                formatted = value_formatter.format(float(value))
                is_bad = tol is not None and abs(float(value)) >= tol
            label = labels[index - 1] if labels is not None else str(index)
            val_color = "#dc2626" if is_bad else "#0f172a"
            bg_color = "background:#fef2f2;" if is_bad else ""
            cells.append(
                f'<td style="padding:6px 10px;border:1px solid #e2e8f0;white-space:nowrap;{bg_color}">'
                f'<div style="font-size:11px;color:#64748b;">{html.escape(label)}</div>'
                f"<div style=\"font-size:13px;color:{val_color};font-weight:{'600' if is_bad else '400'};\">{html.escape(formatted)}</div>"
                "</td>"
            )

        return (
            f'<div style="margin:10px 0 14px 0;"><div style="font-weight:600;color:#0f172a;margin-bottom:6px;">{html.escape(title)}</div>'
            '<div style="overflow-x:auto;"><table style="border-collapse:collapse;"><tr>'
            + "".join(cells)
            + "</tr></table></div></div>"
        )

    @staticmethod
    def _moments_to_html(moments: np.ndarray, variable_names: list[str]) -> str:
        """Render unconditional covariance matrix as an HTML table."""
        import pandas as pd

        df = pd.DataFrame(
            moments,
            index=variable_names,
            columns=variable_names,
        )
        return df.to_html()

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
                parts.append(
                    f'<line x1="{legend_x}" y1="16" x2="{legend_x + 18}" y2="16" stroke="{color}" stroke-width="2"/>'
                )
                parts.append(
                    f'<text x="{legend_x + 24}" y="20" font-size="12" fill="#334155">{safe_shock}</text>'
                )
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
                    parts.append(
                        f'<polyline fill="none" stroke="{color}" stroke-width="2" points="{points}"/>'
                    )

        parts.append("</svg>")
        return "".join(parts)

    def _repr_markdown_(self) -> str | None:
        if str(self.output_type).lower() == "text" and self.mime_bundle_repr is None:
            return None

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

        moments_df = None
        if self.moments is not None and model is not None:
            import pandas as pd

            moments_df = pd.DataFrame(
                self.moments,
                index=model.symbols["endogenous"],
                columns=model.symbols["endogenous"],
            )

        d: dict[str, Any] = {
            "model": model,
            "residuals": self.residuals,
            "dr": dr,
            "sim": self.simulation,
            "fig": self.figure,
            "eigenvalues": self.eigenvalues,
            "moments": self.moments,
            "moments_df": moments_df,
        }
        d.update(context)

        txt = template.substitute(**d)

        for e in unhandled_errors:
            print("Unhandled error:")
            print(e)

        return txt

    # -- Display: HTML (rich console) ----------------------------------------

    def _repr_mimebundle_(
        self,
        include: list[str] | None = None,
        exclude: list[str] | None = None,
    ) -> dict[str, Any]:
        """Return a MIME bundle for notebook-like frontends.

        This keeps the custom Dyno highlighting payload available even when
        frontends select a rich representation from the returned object.
        """
        if self.elapsed is None:
            self.finish()

        data = self._base_mimebundle()

        if include is not None:
            include_set = set(include)
            data = {k: v for k, v in data.items() if k in include_set}

        if exclude is not None:
            exclude_set = set(exclude)
            data = {k: v for k, v in data.items() if k not in exclude_set}

        return data

    def _render_html_report(self) -> str:
        parts: list[str] = []
        if self.model is not None and hasattr(self.model, "_repr_html_"):
            parts.append(self.model._repr_html_())

        check_parts: list[str] = []
        if self.residuals is not None:
            eq_labels: list[str] | None = None
            if self.model is not None and hasattr(self.model, "symbolic"):
                try:
                    eq_labels = [
                        f"eq {i + 1}" for i in range(len(self.model.symbolic.equations))
                    ]
                except Exception:
                    pass
            check_parts.append(
                self._vector_to_horizontal_html(
                    self.residuals,
                    title="Residuals",
                    tol=1e-6,
                    labels=eq_labels,
                )
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
            if self.moments is not None and self.model is not None:
                moments_html = self._moments_to_html(
                    self.moments, self.model.symbols["endogenous"]
                )
                if moments_html:
                    parts.append("<h3>Moments</h3>")
                    parts.append(moments_html)
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

    def _repr_html_(self) -> str | None:
        if self.output_type != "html":
            return None
        return self._render_html_report()

    # -- Display: JupyterLab highlighting ------------------------------------

    @property
    def _highlighting_data(self) -> list[dict[str, Any]]:
        data: list[dict[str, Any]] = []
        for entry in self.errors + self.warnings:
            if "line" in entry:
                item = {
                    "line": entry["line"],
                    "type": entry["type"],
                    "message": entry["message"],
                }
                if "column" in entry:
                    item["column"] = entry["column"]
                data.append(item)
        # Emit warnings for equations whose residual exceeds tolerance
        if (
            self.residuals is not None
            and self.model is not None
            and hasattr(self.model, "symbolic")
        ):
            try:
                eqs = self.model.symbolic.equations
                tol = 1e-6
                for i, (eq, res) in enumerate(
                    zip(eqs, np.asarray(self.residuals).reshape(-1))
                ):
                    if abs(float(res)) >= tol:
                        line = getattr(getattr(eq, "meta", None), "line", None)
                        if line is not None:
                            data.append(
                                {
                                    "line": line,
                                    "type": "warning",
                                    "message": f"Equation {i + 1}: residual = {float(res):.3e}",
                                }
                            )
            except Exception:
                pass
        return data

    def _base_mimebundle(self) -> dict[str, Any]:
        """Return the canonical report MIME payloads shared across frontends."""
        if self.elapsed is None:
            self.finish()

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
            markdown = self._repr_markdown_()
            if markdown:
                data["text/markdown"] = markdown

        include_html = mode == "html" or (mode is None and default_html)
        if include_html:
            html_repr = self._render_html_report()
            if html_repr:
                data["text/html"] = html_repr

        if mode in {None, "text"}:
            data["text/plain"] = repr(self)
        return data

    def jupyter_display(self, *, emit_highlighting: bool = True) -> None:
        # Disabled intentionally: report rendering/display must be controlled by
        # the caller interface, and dsge_report now only emits notifications.
        #
        # Legacy implementation kept commented for future reference:
        # try:
        #     from IPython.display import display, Markdown
        # except ImportError:
        #     self.console_display()
        #     return
        #
        # if self.elapsed is None:
        #     self.finish()
        #
        # _send_interface_notifications(
        #     self,
        #     include_highlighting=emit_highlighting,
        # )
        #
        # display(Markdown(self._repr_markdown_()))
        #
        # if self.figure is not None:
        #     display(self.figure)
        #
        # display(Markdown("---"))
        _ = emit_highlighting
        return

    def display(self) -> None:
        """Display the report in notebook frontends.

        This renders the selected textual representation first, then emits the
        figure object separately so frontends can render rich graph MIME.
        """
        try:
            from IPython.display import display, Markdown, HTML
        except ImportError:
            self.console_display()
            return

        output_mode = str(self.output_type).lower()

        if output_mode == "html":
            html_repr = self._render_html_report()
            if html_repr:
                display(HTML(html_repr))
        elif output_mode == "text":
            display({"text/plain": repr(self)}, raw=True)
        else:
            markdown = self._repr_markdown_()
            if markdown:
                display(Markdown(markdown))

        if self.figure is not None:
            display(self.figure)

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

    @staticmethod
    def _format_symbol_list(names: list[str], *, max_items: int = 10) -> str:
        if not names:
            return "(none)"
        if len(names) <= max_items:
            return ", ".join(names)
        head = ", ".join(names[:max_items])
        return f"{head}, ... (+{len(names) - max_items} more)"

    @staticmethod
    def _format_line_prefix(entry: dict[str, Any]) -> str:
        line = entry.get("line")
        column = entry.get("column")
        if line is None:
            return ""
        if column is None:
            return f"line {line}: "
        return f"line {line}:{column}: "

    def _residuals_summary_lines(self) -> list[str]:
        if self.residuals is None:
            return ["Residuals: not computed"]

        residuals = np.asarray(self.residuals, dtype=float).reshape(-1)
        if residuals.size == 0:
            return ["Residuals: computed (empty)"]

        abs_res = np.abs(residuals)
        max_abs = float(abs_res.max())
        tol = 1e-6
        failing = np.where(abs_res >= tol)[0]

        lines = [
            (
                "Residuals: computed "
                f"(n={residuals.size}, max|r|={max_abs:.3e}, "
                f"nonzero@{tol:.0e}={len(failing)})"
            )
        ]

        if len(failing) == 0:
            return lines

        eq_lines: list[int | None] = []
        if self.model is not None:
            eq_line_getter = getattr(self.model, "_equation_line_numbers", None)
            if callable(eq_line_getter):
                try:
                    eq_lines = list(eq_line_getter())
                except Exception:
                    eq_lines = []

        order = list(np.argsort(abs_res)[::-1][: min(8, residuals.size)])
        lines.append("  largest residuals:")
        for idx in order:
            eq_no = idx + 1
            src_line = eq_lines[idx] if idx < len(eq_lines) else None
            line_txt = f", source line {src_line}" if src_line is not None else ""
            lines.append(
                f"    - eq {eq_no}: r={residuals[idx]:+.3e} (|r|={abs_res[idx]:.3e}{line_txt})"
            )
        return lines

    def _eigenvalues_summary_lines(self) -> list[str]:
        if self.eigenvalues is None:
            return ["Eigenvalues: not computed"]

        evs = np.asarray(self.eigenvalues).reshape(-1)
        if evs.size == 0:
            return ["Eigenvalues: computed (empty)"]

        mod = np.abs(evs)
        unstable = int(np.sum(mod > 1.0))
        unit = int(np.sum(np.isclose(mod, 1.0, atol=1e-8)))
        lines = [
            (
                "Eigenvalues: computed "
                f"(n={evs.size}, min|lambda|={float(mod.min()):.3e}, "
                f"max|lambda|={float(mod.max()):.3e}, "
                f"|lambda|>1: {unstable}, |lambda|≈1: {unit})"
            )
        ]

        bk = self.bk_check
        if bk is True:
            lines.append("Blanchard-Kahn conditions: met")
        elif bk is False:
            lines.append("Blanchard-Kahn conditions: NOT met")
        else:
            lines.append("Blanchard-Kahn conditions: not available")

        return lines

    def _simulation_summary_line(self) -> str:
        if self.simulation is None:
            return "Simulation: not computed"

        if isinstance(self.simulation, dict):
            shocks = list(self.simulation.keys())
            horizon = None
            for value in self.simulation.values():
                try:
                    horizon = len(value)
                    break
                except Exception:
                    continue
            horizon_txt = f", horizon={horizon}" if horizon is not None else ""
            shocks_txt = self._format_symbol_list([str(s) for s in shocks], max_items=6)
            return (
                "Simulation: computed "
                f"(IRFs, shocks={len(shocks)}{horizon_txt})\n"
                f"  shocks: {shocks_txt}"
            )

        try:
            shape = getattr(self.simulation, "shape", None)
            if shape is not None:
                return f"Simulation: computed (tabular shape={shape})"
        except Exception:
            pass

        return f"Simulation: computed ({type(self.simulation).__name__})"

    def _diagnostic_lines(
        self,
        *,
        title: str,
        entries: list[dict[str, Any]],
        max_entries: int = 8,
    ) -> list[str]:
        if not entries:
            return [f"{title}: none"]

        lines = [f"{title}: {len(entries)}"]
        for entry in entries[:max_entries]:
            prefix = self._format_line_prefix(entry)
            lines.append(f"  - {prefix}{entry.get('message', '')}")
        if len(entries) > max_entries:
            lines.append(f"  - ... {len(entries) - max_entries} more")
        return lines

    def __str__(self) -> str:
        if self.elapsed is None:
            self.finish()

        lines: list[str] = ["RunResults", "=========="]

        if self.model is not None:
            symbols = getattr(self.model, "symbols", {})
            variables = list(symbols.get("variables", []))
            endogenous = list(symbols.get("endogenous", []))
            exogenous = list(symbols.get("exogenous", []))
            parameters = list(symbols.get("parameters", []))
            deterministic = bool(getattr(self.model, "is_deterministic", False))

            lines.extend(
                [
                    "Model",
                    "-----",
                    f"name: {getattr(self.model, 'name', None)}",
                    f"filename: {getattr(self.model, 'filename', None)}",
                    f"deterministic: {deterministic}",
                    (
                        "symbols: "
                        f"variables={len(variables)}, "
                        f"endogenous={len(endogenous)}, "
                        f"exogenous={len(exogenous)}, "
                        f"parameters={len(parameters)}"
                    ),
                    f"endogenous: {self._format_symbol_list(endogenous)}",
                    f"exogenous: {self._format_symbol_list(exogenous)}",
                    f"parameters: {self._format_symbol_list(parameters)}",
                    "",
                ]
            )

        lines.extend(["Checks", "------"])
        lines.extend(self._residuals_summary_lines())
        lines.extend(self._eigenvalues_summary_lines())
        lines.append("")

        lines.extend(["Outputs", "-------"])
        lines.append(
            "Solution: computed" if self.solution is not None else "Solution: not computed"
        )
        if self.solution is not None and getattr(self.solution, "decision_rule", None):
            dr = self.solution.decision_rule
            x_shape = getattr(getattr(dr, "X", None), "shape", None)
            y_shape = getattr(getattr(dr, "Y", None), "shape", None)
            s_shape = getattr(getattr(dr, "Σ", None), "shape", None)
            lines.append(f"  decision rule matrices: X{x_shape}, Y{y_shape}, Σ{s_shape}")
        lines.extend(self._simulation_summary_line().split("\n"))
        if self.figure is not None:
            lines.append(f"Figure: available ({type(self.figure).__name__})")
        else:
            lines.append("Figure: not available")
        if self.moments is not None:
            shape = getattr(self.moments, "shape", None)
            lines.append(f"Moments: available{f' (shape={shape})' if shape else ''}")
        else:
            lines.append("Moments: not available")
        lines.append("")

        lines.extend(["Diagnostics", "-----------"])
        lines.extend(self._diagnostic_lines(title="Warnings", entries=self.warnings))
        lines.extend(self._diagnostic_lines(title="Errors", entries=self.errors))
        lines.append("")

        elapsed = self.elapsed if self.elapsed is not None else 0.0
        lines.extend(["Timing", "------", f"elapsed: {elapsed:.3f}s"])

        return "\n".join(lines)

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


def _send_interface_notifications(
    results: RunResults,
    *,
    include_highlighting: bool = True,
) -> None:
    """Emit interface-only notifications (custom MIME payloads only)."""
    try:
        from IPython.display import display
    except ImportError:
        return

    highlighting_key = "application/vnd.jupyterlab-dyno.highlighting+json"
    highlighting = results._highlighting_data
    if include_highlighting and highlighting:
        display({highlighting_key: highlighting}, raw=True)


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
    """Run a model and return a :class:`RunResults` report.

    Parameters
    ----------
    txt:
        Model source text.
    filename:
        Path to a model file (used both to load text and to infer the model
        type from the extension).
    **options:
        Forwarded to the model constructor and run pipeline.
    """

    check_output = options.get("check_output", False)
    output_type = options.get("output_type", "markdown")
    mime_bundle_repr = options.get("mime_bundle_repr", None)
    display_graph = options.get("display_graph", False)
    notify_interface = options.get("notify_interface", True)

    if check_output:
        d: dict[str, Any] = {}
        try:
            exec(txt or "", d, d)  # noqa: S102  — preserved existing behaviour
        except Exception as e:
            results = RunResults(
                source_txt=txt,
                output_type=output_type,
                mime_bundle_repr=mime_bundle_repr,
            )
            results.add_error(str(e))
            return results
        try:
            return d["html"]
        except Exception as e:
            results = RunResults(
                source_txt=txt,
                output_type=output_type,
                mime_bundle_repr=mime_bundle_repr,
            )
            results.add_error(str(e))
            return results

    model: AbstractModel | None = None
    results: RunResults

    try:
        model = _create_model(txt, filename, **options)
        run_output = model.run(default_pipeline=False)
        if not isinstance(run_output, RunResults):
            results = RunResults(
                model=model,
                source_txt=txt,
                output_type=output_type,
                mime_bundle_repr=mime_bundle_repr,
            )
            results.add_error(
                f"Unexpected run() result type: {type(run_output).__name__}"
            )
        else:
            results = run_output
            if results.model is None:
                results.model = model
        results.source_txt = txt
        results.output_type = output_type
        results.mime_bundle_repr = mime_bundle_repr

    except SteadyStateError as e:
        # Steady-state check failed: return a partial report with model info/residuals.
        if model is None:
            model = _create_model(txt, filename, **options)
        results = RunResults(
            model=model,
            source_txt=txt,
            output_type=output_type,
            mime_bundle_repr=mime_bundle_repr,
        )
        results.residuals = e.residuals
        results.add_warning(str(e))
        results.finish()

    except Exception as e:
        results = RunResults(
            model=model,
            source_txt=txt,
            output_type=output_type,
            mime_bundle_repr=mime_bundle_repr,
        )
        results.add_error(
            str(e),
            line=getattr(e, "line", None),
            column=getattr(e, "column", None),
        )
        results.errors[-1]["_exception"] = e

    if notify_interface:
        _send_interface_notifications(results)

    if bool(display_graph) and str(output_type).lower() == "markdown":
        results.display()
    else:
        return results
