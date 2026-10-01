from __future__ import annotations

import html
import time
import os
import re
import numpy as np
import pandas as pd
import tempita

from dyno.errors import ParserError, SteadyStateError
from dyno.svg_theme import theme_svg

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from dyno.model import AbstractModel
    from dyno.simul import SimulationResult
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
{[default moments_uncond_df=None]}
{[default moments_cond_df=None]}
{[default sim_svg=None]}
{[default params_df_html=None]}
{[default steady_df_html=None]}
{[default steady_stats=None]}
{[default residuals_df_html=None]}
{[default eigenvalues_df_html=None]}
{[default should_render_steady=True]}
{[default should_render_check=True]}
{[default should_render_solution=True]}
{[default should_render_simulation_tables=True]}
                            

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

{[endif]}

{[for er in unhandled_errors]}
{[error_callout(er)]}

{[endfor]}

{[if len(parser_errors)>0 or len(unhandled_errors)>0]}
---
{[endif]}


{[if model is not None]}

{[model_overview_markdown]}

:::{dropdown} Calibration
Parameter values
{[ params_df_html ]}
Steady state values
{[ steady_df_html ]}
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

{[if should_render_steady and steady_stats is not None]}

## Steady-state calculation

{[py: _st_ok = bool(steady_stats.get('converged', False))]}
{[if _st_ok]}
:::{tip} Steady-state converged
{[else]}
:::{warning} Steady-state did not converge
{[endif]}
:class: dropdown
- **Algorithm**: `{[ steady_stats.get('algorithm', 'hybr') ]}`
{[if steady_stats.get('iterations') is not None]}
- **Iterations**: {[ steady_stats.get('iterations') ]}
{[endif]}
{[if steady_stats.get('function_evaluations') is not None]}
- **Function evaluations**: {[ steady_stats.get('function_evaluations') ]}
{[endif]}
{[if steady_stats.get('jacobian_evaluations') is not None]}
- **Jacobian evaluations**: {[ steady_stats.get('jacobian_evaluations') ]}
{[endif]}
- **Max residual**: `{[ f"{float(steady_stats.get('max_residual', 0.0)):.3e}" if steady_stats.get('max_residual') is not None else 'N/A' ]}`
- **Tolerance**: `{[ f"{float(steady_stats.get('tolerance', 0.0)):.3e}" if steady_stats.get('tolerance') is not None else 'N/A' ]}`
{[if steady_stats.get('message')]}
- **Message**: {[ steady_stats.get('message') ]}
{[endif]}
:::

---
{[endif]}

{[if should_render_check and (residuals is not None or eigenvalues is not None)]}

## Check

{[if residuals is not None]}
{[py: import numpy as _np; _res_ok = bool(_np.max(_np.abs(residuals)) < 1e-6)]}
{[if _res_ok]}
:::{tip} Residuals are zero
{[else]}
:::{warning} Residuals are not zero
{[endif]}
:class: dropdown
{[ residuals_df_html ]}
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
{[ eigenvalues_df_html ]}
:::
{[endif]}

---

{[endif]}




{[if should_render_solution and dr is not None]}
                    
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
                            

{[if should_render_simulation_tables and sim is not None and model.checks['deterministic']==False]}

## Simulation

{[if moments_uncond_df is not None]}
:::{dropdown} Unconditional Moments
{[to_html_table(moments_uncond_df)]}
:::
{[endif]}

{[if moments_cond_df is not None]}
:::{dropdown} Conditional Moments
{[to_html_table(moments_cond_df)]}
:::
{[endif]}

{[if moments is not None and moments_uncond_df is None and moments_cond_df is None]}
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
                            

{[if sim is not None and model.checks['deterministic']==True]}

## Simulation

{[if moments_uncond_df is not None]}
:::{dropdown} Unconditional Moments
{[to_html_table(moments_uncond_df)]}
:::
{[endif]}

{[if moments_cond_df is not None]}
:::{dropdown} Conditional Moments
{[to_html_table(moments_cond_df)]}
:::
{[endif]}

{[if moments is not None and moments_uncond_df is None and moments_cond_df is None]}
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
                            
{[if sim_svg is not None]}

## Plot

{[sim_svg]}

---
{[endif]}
                            

""",
    delimiters=("{[", "]}"),
)


def error_callout(e: BaseException | str) -> str:
    """MyST error admonition for an unexpected exception or error message.

    The first line of the message is the title, followed by the rest of the
    message. For an exception, the full traceback goes in a collapsed
    "Traceback" dropdown inside the admonition.
    """
    import traceback

    message = str(e).strip() or type(e).__name__
    title, _, rest = message.partition("\n")
    lines = [f"::::{{error}} {title}"]
    if rest.strip():
        lines += ["", rest.strip()]
    if isinstance(e, BaseException):
        trace = "".join(traceback.format_exception(type(e), e, e.__traceback__))
        lines += ["", f":::{{dropdown}} Traceback ({type(e).__name__})", "````text"]
        lines += [trace.rstrip(), "````", ":::"]
    lines.append("::::")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# DataFrame styling helpers for reports & representations
# ---------------------------------------------------------------------------


def _format_table_value(v: Any, fmt_spec: str = "{:.6g}") -> str:
    if v is None:
        return "—"
    try:
        if isinstance(v, (float, np.floating)) and np.isnan(v):
            return "—"
    except Exception:
        pass
    if isinstance(v, (int, np.integer)):
        return str(v)
    if isinstance(v, (float, np.floating)):
        if np.isinf(v):
            return "inf" if v > 0 else "-inf"
        return fmt_spec.format(float(v))
    if isinstance(v, (complex, np.complexfloating)):
        c = complex(v)
        if c.imag == 0:
            return fmt_spec.format(c.real)
        sign = "+" if c.imag >= 0 else "-"
        return f"{fmt_spec.format(c.real)} {sign} {fmt_spec.format(abs(c.imag))}j"
    return str(v)


def _inline_styler_styles(html_str: str) -> str:
    style_match = re.search(r"<style[^>]*>(.*?)</style>", html_str, re.DOTALL)
    if not style_match:
        return html_str
    css_content = style_match.group(1)
    rule_pattern = re.compile(r"([^{]+)\{([^}]+)\}")
    id_to_style: dict[str, str] = {}
    for match in rule_pattern.finditer(css_content):
        selectors, declarations = match.groups()
        cleaned_style = re.sub(r"\s+", " ", declarations).strip()
        for sel in selectors.split(","):
            sel_id = sel.strip().lstrip("#")
            if sel_id:
                if sel_id in id_to_style:
                    id_to_style[sel_id] = id_to_style[sel_id] + "; " + cleaned_style
                else:
                    id_to_style[sel_id] = cleaned_style

    def replace_tag(tag_match: re.Match[str]) -> str:
        tag = tag_match.group(0)
        id_m = re.search(r'id=["\']([^"\']+)["\']', tag)
        if id_m:
            elem_id = id_m.group(1)
            if elem_id in id_to_style:
                style_val = id_to_style[elem_id]
                if 'style="' in tag:
                    tag = tag.replace('style="', f'style="{style_val}; ')
                elif "style='" in tag:
                    tag = tag.replace("style='", f"style='{style_val}; ")
                else:
                    tag = tag[:-1] + f' style="{style_val}">'
        return tag

    return re.sub(r"<(?:td|th)\b[^>]*>", replace_tag, html_str)


def style_dataframe(df: pd.DataFrame, *, hide_index: bool = False) -> Any:
    s = df.style.format(lambda v: _format_table_value(v, "{:.6g}"))
    if hide_index:
        s = s.hide(axis="index")
    return s


def style_calibration_dataframe(df: pd.DataFrame, *, hide_index: bool = False) -> Any:
    def highlight(val: Any) -> str:
        try:
            if isinstance(val, (float, np.floating)) and np.isnan(val):
                return "background-color: #fef2f2; color: #dc2626; font-weight: 600;"
        except Exception:
            if val is None:
                return "background-color: #fef2f2; color: #dc2626; font-weight: 600;"
        return ""

    s = df.style.map(highlight).format(lambda v: _format_table_value(v, "{:.6g}"))
    if hide_index:
        s = s.hide(axis="index")
    return s


def style_residuals_dataframe(
    df: pd.DataFrame, *, tol: float = 1e-6, hide_index: bool = False
) -> Any:
    def highlight(val: Any) -> str:
        try:
            f = float(val)
            if np.isnan(f) or abs(f) >= tol:
                return "background-color: #fef2f2; color: #dc2626; font-weight: 600;"
        except (ValueError, TypeError):
            pass
        return ""

    s = df.style.map(highlight).format(lambda v: _format_table_value(v, "{:.4e}"))
    if hide_index:
        s = s.hide(axis="index")
    return s


def style_eigenvalues_dataframe(
    df: pd.DataFrame, *, n_eq: int | None = None, hide_index: bool = False
) -> Any:
    if n_eq is None:
        if len(df.columns) > 1 or (len(df) == 1 and len(df.columns) > 0):
            n_eq = len(df.columns) // 2
        else:
            n_eq = len(df.index) // 2

    is_horizontal = len(df.columns) >= len(df.index)

    def highlight_eigenvalues(data: pd.DataFrame) -> pd.DataFrame:
        styles = pd.DataFrame("", index=data.index, columns=data.columns)
        if is_horizontal:
            for col_idx, col in enumerate(data.columns):
                i = col_idx + 1  # 1-indexed
                border = (
                    "border-right: 2px solid #94a3b8;" if (n_eq and i == n_eq) else ""
                )
                for row in data.index:
                    val = data.loc[row, col]
                    try:
                        mod = abs(complex(val))
                        if np.isnan(mod):
                            styles.loc[row, col] = (
                                f"background-color: #fef2f2; color: #dc2626; font-weight: 600; {border}".strip()
                            )
                        elif (n_eq and i <= n_eq and mod > 1.0) or (
                            n_eq and i > n_eq and mod <= 1.0
                        ):
                            styles.loc[row, col] = (
                                f"background-color: #fef2f2; color: #dc2626; font-weight: 600; {border}".strip()
                            )
                        elif border:
                            styles.loc[row, col] = border
                    except Exception:
                        if border:
                            styles.loc[row, col] = border
        else:
            for row_idx, row in enumerate(data.index):
                i = row_idx + 1  # 1-indexed
                border = (
                    "border-bottom: 2px solid #94a3b8;" if (n_eq and i == n_eq) else ""
                )
                for col in data.columns:
                    val = data.loc[row, col]
                    try:
                        mod = abs(complex(val))
                        if np.isnan(mod):
                            styles.loc[row, col] = (
                                f"background-color: #fef2f2; color: #dc2626; font-weight: 600; {border}".strip()
                            )
                        elif (n_eq and i <= n_eq and mod > 1.0) or (
                            n_eq and i > n_eq and mod <= 1.0
                        ):
                            styles.loc[row, col] = (
                                f"background-color: #fef2f2; color: #dc2626; font-weight: 600; {border}".strip()
                            )
                        elif border:
                            styles.loc[row, col] = border
                    except Exception:
                        if border:
                            styles.loc[row, col] = border
        return styles

    s = df.style.apply(highlight_eigenvalues, axis=None).format(
        lambda v: _format_table_value(v, "{:.4g}")
    )
    if hide_index:
        s = s.hide(axis="index")
    return s


# ---------------------------------------------------------------------------
# RunResults — unified result container for all model execution paths
# ---------------------------------------------------------------------------


def _embed_svg(svg: str, output_type: str, alt: str) -> str:
    """Embed an SVG chart in the markdown report.

    With `output_type="myst"` dyno renders the report itself, so the SVG is
    inlined (on one line, inside a `<div>`, so markdown passes it through)
    and follows the page theme (see `dyno.svg_theme`). Other markdown
    renderers get a self-contained `<img>`, in light colors.
    """
    if str(output_type).lower() == "myst":
        return f'<div class="dyno-plot">{svg}</div>'
    import base64

    b64 = base64.b64encode(svg.encode("utf-8")).decode("ascii")
    return f'<img src="data:image/svg+xml;base64,{b64}" alt="{alt}" style="max-width:100%; height:auto;" />'


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
        output_type: str = "html",
        mime_bundle_repr: str | None = None,
    ) -> None:
        self.model: AbstractModel | None = model
        self.source_txt: str | None = source_txt
        # Default to HTML rendering for richer notebook output.
        self.output_type: str = output_type
        # MIME bundle policy: None => include all rich reprs; else one of
        # {"markdown", "html", "text"} to include only that representation.
        self.mime_bundle_repr: str | None = mime_bundle_repr

        # Pipeline outputs
        self.steady_stats: dict[str, Any] | None = None
        if model is not None and getattr(model, "steady_stats", None) is not None:
            self.steady_stats = model.steady_stats
        self.residuals: np.ndarray | None = None
        self.solution: PerturbationSolution | Any | None = None
        self.simulation: SimulationResult | dict | pd.DataFrame | None = None
        self.figure: Any | None = None
        self.eigenvalues: np.ndarray | None = None
        self.moments: np.ndarray | None = None

        # Structured diagnostics: list of {line, type, message}
        self.errors: list[dict[str, Any]] = []
        self.warnings: list[dict[str, Any]] = []

        # Timing
        self._t_start: float = time.time()
        self.elapsed: float | None = None
        self._from_pipeline: bool = False
        self._plot_options: dict[str, Any] | None = None
        self.muted_commands: set[str] = set()

    @property
    def steady_info(self) -> dict[str, Any] | None:
        """Alias for steady_stats."""
        return self.steady_stats

    @steady_info.setter
    def steady_info(self, value: dict[str, Any] | None) -> None:
        self.steady_stats = value

    @property
    def _should_render_steady(self) -> bool:
        if self.steady_stats is None:
            return False
        if "steady" in self.muted_commands:
            return not bool(self.steady_stats.get("converged", True))
        return True

    def _is_check_error(self) -> bool:
        if self.residuals is not None:
            try:
                if np.max(np.abs(self.residuals)) >= 1e-6:
                    return True
            except Exception:
                pass
        if self.eigenvalues is not None:
            try:
                evs_mod = np.abs(self.eigenvalues)
                n = len(evs_mod) // 2
                if n > 0:
                    bk_met = bool(evs_mod[n - 1] < 1 < evs_mod[n])
                    if not bk_met:
                        return True
            except Exception:
                pass
        for e in self.errors:
            msg = str(e.get("message", "")).lower()
            if any(
                term in msg for term in ("check", "residual", "eigenvalue", "blanchard")
            ):
                return True
        for w in self.warnings:
            msg = str(w.get("message", "")).lower()
            if any(
                term in msg for term in ("check", "residual", "eigenvalue", "blanchard")
            ):
                return True
        return False

    @property
    def _should_render_check(self) -> bool:
        if "check" in self.muted_commands:
            return self._is_check_error()
        return True

    def _is_solution_error(self) -> bool:
        if self.solution is None:
            return True
        for e in self.errors:
            msg = str(e.get("message", "")).lower()
            if "solve" in msg or "solution" in msg:
                return True
        return False

    @property
    def _should_render_solution(self) -> bool:
        if "solve" in self.muted_commands or "perturb" in self.muted_commands:
            return self._is_solution_error()
        return True

    @property
    def _should_render_simulation_tables(self) -> bool:
        if "simulate" in self.muted_commands or "simul" in self.muted_commands:
            return False
        return True

    @property
    def _should_render_plot(self) -> bool:
        if "plot" in self.muted_commands:
            return False
        if self._from_pipeline:
            return self.figure is not None
        return self.simulation is not None or self.figure is not None

    # -- Diagnostic helpers --------------------------------------------------

    def add_error(
        self,
        message: str,
        *,
        line: int | None = None,
        column: int | None = None,
    ) -> None:
        if line is None:
            m_line = re.search(r"\blines?\s+(\d+)", message, flags=re.IGNORECASE)
            if m_line is not None:
                line = int(m_line.group(1))
        if column is None:
            m_col = re.search(r"\bcols?\s+(\d+)", message, flags=re.IGNORECASE)
            if m_col is not None:
                column = int(m_col.group(1))

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
            return _inline_styler_styles(value.to_html())
        if hasattr(value, "to_frame"):
            return _inline_styler_styles(value.to_frame().to_html())
        return f"<pre>{value}</pre>"

    def parameters_dataframe(
        self, orientation: str = "horizontal"
    ) -> pd.DataFrame | None:
        if self.model is None:
            return None
        import pandas as pd

        c_dict = self.model.context.get("constants", {})
        params = self.model.symbols.get("parameters", list(c_dict.keys()))
        row = {p: c_dict.get(p, np.nan) for p in params}
        df = pd.DataFrame([row], index=[getattr(self.model, "name", None) or "Value"])
        return df if orientation == "horizontal" else df.T

    def steady_state_dataframe(
        self, orientation: str = "horizontal"
    ) -> pd.DataFrame | None:
        if self.model is None:
            return None
        import pandas as pd

        s_dict = self.model.context.get("steady_states", {})
        variables = self.model.symbols.get("variables", list(s_dict.keys()))
        row = {v: s_dict.get(v, np.nan) for v in variables}
        df = pd.DataFrame([row], index=[getattr(self.model, "name", None) or "Value"])
        return df if orientation == "horizontal" else df.T

    def residuals_dataframe(
        self, orientation: str = "horizontal"
    ) -> pd.DataFrame | None:
        if self.residuals is None:
            return None
        import pandas as pd

        if isinstance(self.residuals, pd.Series):
            arr = self.residuals.values
            eq_labels = list(self.residuals.index.astype(str))
        else:
            arr = np.asarray(self.residuals, dtype=float).reshape(-1)
            eq_labels = [f"eq {i + 1}" for i in range(len(arr))]
        df = pd.DataFrame(
            [arr],
            index=[getattr(self.model, "name", None) or "Value"],
            columns=eq_labels,
        )
        return df if orientation == "horizontal" else df.T

    def eigenvalues_dataframe(
        self, orientation: str = "horizontal"
    ) -> pd.DataFrame | None:
        if self.eigenvalues is None:
            return None
        import pandas as pd

        if isinstance(self.eigenvalues, pd.Series):
            arr = self.eigenvalues.values
            ev_labels = list(self.eigenvalues.index.astype(str))
        else:
            arr = np.asarray(self.eigenvalues).reshape(-1)
            ev_labels = [str(i + 1) for i in range(len(arr))]
        df = pd.DataFrame(
            [arr],
            index=[getattr(self.model, "name", None) or "Value"],
            columns=ev_labels,
        )
        return df if orientation == "horizontal" else df.T

    @property
    def parameters_df(self) -> pd.DataFrame | None:
        return self.parameters_dataframe()

    @property
    def steady_state_df(self) -> pd.DataFrame | None:
        return self.steady_state_dataframe()

    @property
    def residuals_df(self) -> pd.DataFrame | None:
        return self.residuals_dataframe()

    @property
    def eigenvalues_df(self) -> pd.DataFrame | None:
        return self.eigenvalues_dataframe()

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
        """Render a covariance matrix as an HTML table."""
        import pandas as pd

        df = pd.DataFrame(
            moments,
            index=variable_names,
            columns=variable_names,
        )
        return df.to_html()

    def _moments_dataframes(
        self,
    ) -> tuple["pd.DataFrame | None", "pd.DataFrame | None"]:
        """Return (conditional_df, unconditional_df) when available."""
        if self.model is None:
            return None, None

        import pandas as pd

        names = self.model.symbols["endogenous"]
        conditional_df = None
        unconditional_df = None

        if self.solution is not None and hasattr(self.solution, "moments"):
            try:
                conditional, unconditional = self.solution.moments()
                conditional_df = pd.DataFrame(
                    conditional,
                    index=names,
                    columns=names,
                )
                unconditional_df = pd.DataFrame(
                    unconditional,
                    index=names,
                    columns=names,
                )
            except Exception:
                pass

        if unconditional_df is None and self.moments is not None:
            unconditional_df = pd.DataFrame(
                self.moments,
                index=names,
                columns=names,
            )

        return conditional_df, unconditional_df

    @staticmethod
    def _simulation_to_html(simulation: Any, variables: list[str] | None = None) -> str:
        import pandas as pd

        from dyno.simul import SimulationResult, sim_to_nsim

        if isinstance(simulation, (SimulationResult, dict)):
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

        all_variables = list(dict.fromkeys(data["variable"].astype(str)))
        if variables is not None:
            variables_list = [str(v) for v in variables if str(v) in all_variables]
        else:
            variables_list = all_variables
        if not variables_list:
            return ""
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
        rows = max(1, (len(variables_list) + cols - 1) // cols)
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

        for index, variable in enumerate(variables_list):
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
        return theme_svg("".join(parts))

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
        # Other exceptions, and errors recorded as plain messages
        unhandled_errors = [
            e.get("_exception", e["message"])
            for e in self.errors
            if not isinstance(e.get("_exception"), ParserError)
        ]
        error_lines = [
            str(e.line)
            for e in parser_errors
            if hasattr(e, "line") and e.line is not None
        ]

        context: dict[str, Any] = {
            "traceback": tb_mod,
            "errors": [
                e.get("_exception") for e in exception_errors if "_exception" in e
            ],
            "parser_errors": parser_errors,
            "unhandled_errors": unhandled_errors,
            "error_callout": error_callout,
            "error_lines": error_lines,
            "alt": altair,
            "to_html_table": self._to_html_table,
        }

        model = self.model
        params_df_html: str | None = None
        steady_df_html: str | None = None
        if model is not None:
            from dyno.model_render import (
                model_repr_data,
                render_model_overview_markdown,
            )

            p_df = self.parameters_dataframe(orientation="horizontal")
            if p_df is not None:
                params_df_html = _inline_styler_styles(
                    style_calibration_dataframe(p_df, hide_index=True).to_html()
                )

            s_df = self.steady_state_dataframe(orientation="horizontal")
            if s_df is not None:
                steady_df_html = _inline_styler_styles(
                    style_calibration_dataframe(s_df, hide_index=True).to_html()
                )

            context["model_overview_markdown"] = render_model_overview_markdown(
                model_repr_data(model), model.filename
            )

        residuals_df_html: str | None = None
        if self.residuals is not None:
            r_df = self.residuals_dataframe(orientation="horizontal")
            if r_df is not None:
                residuals_df_html = _inline_styler_styles(
                    style_residuals_dataframe(r_df, hide_index=True).to_html()
                )

        eigenvalues_df_html: str | None = None
        if self.eigenvalues is not None:
            ev_df = self.eigenvalues_dataframe(orientation="horizontal")
            if ev_df is not None:
                n_eq = (
                    len(self.model.symbols.get("endogenous", []))
                    if self.model and "endogenous" in getattr(self.model, "symbols", {})
                    else len(ev_df.columns) // 2
                )
                eigenvalues_df_html = _inline_styler_styles(
                    style_eigenvalues_dataframe(
                        ev_df, n_eq=n_eq, hide_index=True
                    ).to_html()
                )

        context["params_df_html"] = params_df_html
        context["steady_df_html"] = steady_df_html
        context["residuals_df_html"] = residuals_df_html
        context["eigenvalues_df_html"] = eigenvalues_df_html

        dr = self.solution
        if dr is not None:
            bk = self.bk_check
            context["bk_check"] = bk
            context["jacs"] = dr.coefficients_as_df()

        moments_cond_df, moments_uncond_df = self._moments_dataframes()
        moments_df = moments_uncond_df

        sim_svg: str | None = None
        if (
            self._should_render_plot
            and self._from_pipeline
            and self.figure is not None
            and self.simulation is not None
        ):
            plot_vars = (
                (self._plot_options or {}).get("variables")
                if isinstance(self._plot_options, dict)
                else None
            )
            rendered_svg = self._simulation_to_html(
                self.simulation, variables=plot_vars
            )
            if rendered_svg:
                sim_svg = _embed_svg(
                    rendered_svg, self.output_type, "Simulation charts"
                )

        d: dict[str, Any] = {
            "model": model,
            "residuals": self.residuals,
            "dr": dr,
            "sim": self.simulation,
            "fig": self.figure,
            "sim_svg": sim_svg,
            "eigenvalues": self.eigenvalues,
            "moments": self.moments,
            "moments_df": moments_df,
            "moments_cond_df": moments_cond_df,
            "moments_uncond_df": moments_uncond_df,
            "steady_stats": self.steady_stats,
            "should_render_steady": self._should_render_steady,
            "should_render_check": self._should_render_check,
            "should_render_solution": self._should_render_solution,
            "should_render_simulation_tables": self._should_render_simulation_tables,
        }
        d.update(context)

        txt = template.substitute(**d)

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

    def _steady_stats_to_html(self) -> str:
        if self.steady_stats is None:
            return ""
        stats = self.steady_stats
        algo = html.escape(str(stats.get("algorithm", "hybr")))
        converged = bool(stats.get("converged", False))
        status_badge = (
            '<span style="color:#16a34a;font-weight:600;">Converged</span>'
            if converged
            else '<span style="color:#dc2626;font-weight:600;">Did not converge</span>'
        )
        max_res = stats.get("max_residual")
        max_res_str = (
            f"{float(max_res):.3e}"
            if max_res is not None and np.isfinite(max_res)
            else "N/A"
        )
        tol = stats.get("tolerance")
        tol_str = f"{float(tol):.3e}" if tol is not None else "N/A"

        rows: list[tuple[str, str]] = [
            ("Algorithm", algo),
            ("Status", status_badge),
        ]
        if stats.get("iterations") is not None:
            rows.append(("Iterations", html.escape(str(stats["iterations"]))))
        if stats.get("function_evaluations") is not None:
            rows.append(
                (
                    "Function evaluations",
                    html.escape(str(stats["function_evaluations"])),
                )
            )
        if stats.get("jacobian_evaluations") is not None:
            rows.append(
                (
                    "Jacobian evaluations",
                    html.escape(str(stats["jacobian_evaluations"])),
                )
            )
        rows.append(("Max residual", html.escape(max_res_str)))
        rows.append(("Tolerance", html.escape(tol_str)))
        if stats.get("message"):
            rows.append(("Message", html.escape(str(stats["message"]))))

        cells = "".join(
            f'<tr><td style="padding:6px 12px;font-weight:600;color:#475569;border:1px solid #e2e8f0;background:#f8fafc;">{label}</td>'
            f'<td style="padding:6px 12px;border:1px solid #e2e8f0;">{val}</td></tr>'
            for label, val in rows
        )

        border_color = "#86efac" if converged else "#fca5a5"
        bg_header = "#f0fdf4" if converged else "#fef2f2"
        text_color = "#166534" if converged else "#991b1b"
        summary_title = (
            "Steady-state converged" if converged else "Steady-state did not converge"
        )

        return (
            '<div style="margin:12px 0;">'
            "<h3>Steady-state calculation</h3>"
            f'<details style="border:1px solid {border_color};border-radius:6px;background:#ffffff;margin:8px 0;" open>'
            f'<summary style="background:{bg_header};color:{text_color};padding:8px 12px;font-weight:600;cursor:pointer;">{summary_title}</summary>'
            f'<div style="padding:10px 12px;overflow-x:auto;">'
            f'<table style="border-collapse:collapse;font-size:13px;border:1px solid #cbd5e1;">'
            f"<tbody>{cells}</tbody>"
            f"</table>"
            f"</div>"
            f"</details>"
            "</div>"
        )

    def _render_html_report(self) -> str:
        parts: list[str] = []
        if self.model is not None and hasattr(self.model, "_repr_html_"):
            parts.append(self.model._repr_html_())

        if self._should_render_steady:
            parts.append(self._steady_stats_to_html())

        check_parts: list[str] = []
        if self._should_render_check:
            if self.residuals is not None:
                eq_labels: list[str] | None = None
                if self.model is not None and hasattr(self.model, "symbolic"):
                    try:
                        eq_labels = [
                            f"eq {i + 1}"
                            for i in range(len(self.model.symbolic.equations))
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

        if (
            self._should_render_solution
            and self.solution is not None
            and hasattr(self.solution, "_repr_html_")
        ):
            parts.append(self.solution._repr_html_())

        if self.simulation is not None:
            if self._should_render_simulation_tables:
                moments_cond_df, moments_uncond_df = self._moments_dataframes()
                if moments_cond_df is not None or moments_uncond_df is not None:
                    parts.append("<h3>Moments</h3>")
                    if moments_uncond_df is not None:
                        parts.append("<h4>Unconditional Moments</h4>")
                        parts.append(moments_uncond_df.to_html())
                    if moments_cond_df is not None:
                        parts.append("<h4>Conditional Moments</h4>")
                        parts.append(moments_cond_df.to_html())
            if self._should_render_plot:
                plot_vars = (
                    (self._plot_options or {}).get("variables")
                    if isinstance(self._plot_options, dict)
                    else None
                )
                sim_html = self._simulation_to_html(
                    self.simulation, variables=plot_vars
                )
                if sim_html:
                    parts.append("<h3>Simulation</h3>")
                    parts.append(sim_html)
                elif self.figure is not None:
                    parts.append("<h3>Simulation</h3>")
                    parts.append(self._figure_to_html(self.figure))
        elif self.figure is not None and self._should_render_plot:
            parts.append("<h3>Simulation</h3>")
            parts.append(self._figure_to_html(self.figure))
        for e in self.errors:
            parts.append(f"<pre style='color:red'>{e['message']}</pre>")
        return "<br>".join(parts)

    @staticmethod
    def _render_myst_html(markdown: str) -> str:
        """Render the MyST report to HTML ourselves, so it displays without a
        frontend MyST renderer (e.g. on JupyterLite, where `jupyterlab-myst`
        can't be installed)."""
        from dyno.myst import render_markdown_myst

        return render_markdown_myst(markdown)

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
                if output_mode == "myst":
                    data["text/html"] = self._render_myst_html(markdown)

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
        elif output_mode == "myst":
            markdown = self._repr_markdown_()
            if markdown:
                display(HTML(self._render_myst_html(markdown)))
        else:
            markdown = self._repr_markdown_()
            if markdown:
                display(Markdown(markdown))

        if self.figure is not None and (
            not self._from_pipeline or self.simulation is None
        ):
            display(self.figure)

    def console_display(self) -> None:
        if self.elapsed is None:
            self.finish()

        if self.model is not None:
            print(repr(self.model))

        if self._should_render_steady and self.steady_stats is not None:
            algo = self.steady_stats.get("algorithm", "hybr")
            conv = "converged" if self.steady_stats.get("converged") else "failed"
            print(f"Steady-state calculation: {algo} ({conv})")

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
            plot_txt = self.plot_text(color=True)
            if plot_txt:
                print(plot_txt)

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

    def _steady_summary_lines(self) -> list[str]:
        if self.steady_stats is None:
            return []
        stats = self.steady_stats
        algo = stats.get("algorithm", "hybr")
        converged = bool(stats.get("converged", False))
        status_str = "converged" if converged else "failed to converge"
        max_res = stats.get("max_residual")
        max_res_str = (
            f"{float(max_res):.3e}"
            if max_res is not None and np.isfinite(max_res)
            else "N/A"
        )
        tol = stats.get("tolerance")
        tol_str = f"{float(tol):.3e}" if tol is not None else "N/A"

        lines = [
            "Steady-state calculation",
            "------------------------",
            f"algorithm: {algo}",
            f"status: {status_str}",
        ]
        if stats.get("iterations") is not None:
            lines.append(f"iterations: {stats['iterations']}")
        if stats.get("function_evaluations") is not None:
            lines.append(f"function evaluations: {stats['function_evaluations']}")
        if stats.get("jacobian_evaluations") is not None:
            lines.append(f"jacobian evaluations: {stats['jacobian_evaluations']}")
        lines.append(f"max residual: {max_res_str}")
        lines.append(f"tolerance: {tol_str}")
        if stats.get("message"):
            lines.append(f"message: {stats['message']}")
        lines.append("")
        return lines

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

    def plot_text(
        self,
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
        """Render simulation / IRF graphs as text using plotext.

        Parameters
        ----------
        cols : int, default 2
            Number of subplot columns.
        width : int | None, optional
            Total width in characters. If None, detected from terminal or defaults to 80.
        height : int | None, optional
            Total height in lines. If None, automatically scaled to rows * 10.
        variables : list[str] | None, optional
            List of variable names to plot. If None, plots all variables in simulation.
        color : bool | None, optional
            Whether to retain ANSI color codes. If None, detected from stdout.
        theme : str, default "clear"
            Plotext theme.
        marker : str | None, optional
            Marker style (e.g. "hd", "braille", "sd", "dot").
        show : bool, default False
            If True, prints the plot to stdout.

        Returns
        -------
        str
            The rendered text plot.
        """
        if self.simulation is None:
            return ""
        from dyno.plots import plot_simulation_plotext

        if variables is None and isinstance(self._plot_options, dict):
            variables = self._plot_options.get("variables")

        return plot_simulation_plotext(
            self.simulation,
            cols=cols,
            width=width,
            height=height,
            variables=variables,
            color=color,
            theme=theme,
            marker=marker,
            show=show,
        )

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
        """Return the formatted plain-text diagnostic and results summary.

        Parameters
        ----------
        graphs : bool | None, optional
            Whether to include plotext ASCII/ANSI graphs of simulations/IRFs.
            If None, renders graphs when `_should_render_plot` is True.
        color : bool | None, optional
            Whether to include ANSI colors in graphs. If None, detected from stdout.
        width : int | None, optional
            Width for graph rendering.
        height : int | None, optional
            Height for graph rendering.
        cols : int, default 2
            Columns for graph layout.
        variables : list[str] | None, optional
            Subset of variables to plot.
        marker : str | None, optional
            Marker style (e.g. "hd", "braille", "sd").
        """
        if self.elapsed is None:
            self.finish()

        lines: list[str] = ["RunResults", "=========="]

        if self.model is not None:
            symbols = getattr(self.model, "symbols", {})
            variables_all = list(symbols.get("variables", []))
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
                        f"variables={len(variables_all)}, "
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

        if self._should_render_steady:
            lines.extend(self._steady_summary_lines())

        if self._should_render_check:
            lines.extend(["Checks", "------"])
            lines.extend(self._residuals_summary_lines())
            lines.extend(self._eigenvalues_summary_lines())
            lines.append("")

        lines.extend(["Outputs", "-------"])
        if self._should_render_solution:
            lines.append(
                "Solution: computed"
                if self.solution is not None
                else "Solution: not computed"
            )
            if self.solution is not None and getattr(
                self.solution, "decision_rule", None
            ):
                dr = self.solution.decision_rule
                x_shape = getattr(getattr(dr, "X", None), "shape", None)
                y_shape = getattr(getattr(dr, "Y", None), "shape", None)
                s_shape = getattr(getattr(dr, "Σ", None), "shape", None)
                lines.append(
                    f"  decision rule matrices: X{x_shape}, Y{y_shape}, Σ{s_shape}"
                )
        if self._should_render_simulation_tables:
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
                lines.extend(
                    [
                        "Simulation Plots",
                        "----------------",
                        sim_plot,
                        "",
                    ]
                )

        lines.extend(["Diagnostics", "-----------"])
        lines.extend(self._diagnostic_lines(title="Warnings", entries=self.warnings))
        lines.extend(self._diagnostic_lines(title="Errors", entries=self.errors))
        lines.append("")

        elapsed = self.elapsed if self.elapsed is not None else 0.0
        lines.extend(["Timing", "------", f"elapsed: {elapsed:.3f}s"])

        return "\n".join(lines)

    def to_markdown(self) -> str:
        """Return the formatted Markdown report."""
        md = self._repr_markdown_()
        return md if md is not None else ""

    def to_html(self) -> str:
        """Return the formatted HTML report."""
        return self._render_html_report()

    def __str__(self) -> str:
        return self.to_text()

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
        preprocessor = (
            options.get("modfile-preprocessor")
            or options.get("modfile_preprocessor")
            or options.get("preprocessor")
            or "dynare"
        )
        if preprocessor == "dynare":
            from dyno.dynare import DynareModel

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
    output_type = options.get("output_type", "html")
    mime_bundle_repr = options.get("mime_bundle_repr", None)
    notify_interface = options.get("notify_interface", True)
    results: RunResults

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

    try:
        model = _create_model(txt, filename, **options)
        from .variants import RunResultsVariants

        run_output = model.run(default_pipeline=False)
        if not isinstance(run_output, (RunResults, RunResultsVariants)):
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
            results = run_output  # type: ignore[assignment]
            if not isinstance(results, RunResultsVariants) and results.model is None:
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
        line = getattr(e, "line", None)
        if line is None:
            line = getattr(e, "begin_line", None)
        column = getattr(e, "column", None)
        if column is None:
            column = getattr(e, "begin_column", None)
        loc = getattr(e, "location", None)
        if loc is not None:
            if line is None:
                line = getattr(loc, "line", getattr(loc, "begin_line", None))
            if column is None:
                column = getattr(loc, "column", getattr(loc, "begin_column", None))
        if line is None and getattr(e, "__cause__", None) is not None:
            cause = e.__cause__
            line = getattr(cause, "line", getattr(cause, "begin_line", None))
            if column is None:
                column = getattr(cause, "column", getattr(cause, "begin_column", None))

        results.add_error(
            str(e),
            line=line,
            column=column,
        )
        results.errors[-1]["_exception"] = e

    if notify_interface:
        _send_interface_notifications(results)

    if str(output_type).lower() in {"markdown", "myst"}:
        results.display()

    return results
