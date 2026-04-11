from __future__ import annotations

import time
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

{[jacs[0].to_html()]}
                            
### Jacobian 

{[jacs[1].to_html()]}

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
{[sim[k].to_html()]}
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
{[sim[k].to_html()]}
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
        error_lines = [
            str(e.line) for e in parser_errors if hasattr(e, "line")
        ]

        context: dict[str, Any] = {
            "traceback": tb_mod,
            "errors": [e.get("_exception") for e in exception_errors if "_exception" in e],
            "parser_errors": parser_errors,
            "unhandled_errors": unhandled_errors,
            "error_lines": error_lines,
            "alt": altair,
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
        if self.residuals is not None:
            parts.append(f"<pre>Residuals: {self.residuals}</pre>")
        if self.solution is not None and hasattr(self.solution, "_repr_html_"):
            parts.append(self.solution._repr_html_())
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
                {
                    "application/vnd.jupyterlab-dyno.highlighting+json": highlighting
                },
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
    txt: str | None, filename: str | None, **options
) -> "AbstractModel":
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
            from dyno.dynare_model import DynareModel as Model
        else:
            from dyno.symbolic_model import DynoModel as Model
        return Model(filename=filename, txt=txt)
    elif filename.endswith(".dyno.yaml"):
        from dyno.yamlfile import YAMLFile
        return YAMLFile(txt=txt)
    elif filename.endswith(".dyno"):
        from dyno.symbolic_model import DynoModel

        return DynoModel(filename=filename, txt=txt)
    else:
        raise ValueError("Unsupported Model type")


def dsge_report(txt: str | None = None, filename: str | None = None, **options) -> RunResults:

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
