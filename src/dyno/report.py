from IPython.display import display, HTML, Markdown, TextDisplayObject
import time
import rich
from rich import inspect
import pandas
import altair
import tempita

from dyno.errors import ParserError

import altair as alt
# load a simple dataset as a pandas DataFrame
from vega_datasets import data



# :::{dropdown} Code
# ```{code}
# :linenos:
# :emphasize-lines: {[ str.join(",",[str(e.token.line) for e in errors]) ]}
# {[code]}
# ```
# :::

# {[if len(errors)>0]}      

# {[for e in errors]}
                                     
# :::{error} UnexpectedToken `{[e.token.value]}` at {[(e.token.line, e.token.column)]}
# :class: dropdown
# ```
# {[str(e.get_context(text=code))]}
# {[str(e)]}
# ```
# :::
                            
# {[endfor]} 
# {[endif]}


template = tempita.Template(r"""
{[default model=None]}
{[default dr=None]}
{[default bk_check=None]}
{[default sim=None]}
{[default fig=None]}                        
                            
# Dyno Report
                            

                            
:::{dropdown} Code
```{code}
:linenos:
:emphasize-lines: {[ str.join(",",error_lines) ]}
{[code]}
```
:::


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

---

{[if model is not None]}
                            
## Model Summary

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

                                                   
{[if not abs(residuals).max() < 1e-6]}
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
""", delimiters=('{[', ']}'))


class Report:

    def __init__(self, *elements, **options):

        self.elements = {}
        self.output_type = options.get("output_type", "markdown")
        self.t_start = time.time()

    def __enter__(self):
        pass

    def __exit__(self, type, value, traceback):

        self("Elapsed time: {:.3f} sec".format(time.time() - self.t_start))


    # def __repr__(self):

    #     self("Elapsed time: {:.3f} sec".format(time.time() - self.t_start))

    #     if self.output_type != "text":
    #         return None
        

    def _repr_html_(self):
        import time


        if self.output_type == "text":
            import rich
            from rich.console import Console
            from rich import print, inspect
            import os
            console = Console(record=True, file=open(os.devnull, "wt"), color_system="truecolor", width=100)
            for k,w in self.elements.items():
                if isinstance(k,int):
                    console.print("---")
                else:
                    console.print("[bold magenta]=== {} ===[/]".format(k))
                console.print(w) 

            return console.export_html(inline_styles=True)
        
        elif self.output_type != "html":
            return None
        
        reprs = []
        for r in self.elements.values():
            if hasattr(r, "_repr_html_"):
                reprs.append(r._repr_html_())
            else:
                reprs.append(f"<pre>{str(r)}</pre>")

        html = "<br>".join([str(e) for e in reprs])

        return html
    
    def _repr_markdown_(self):

        self("Elapsed time: {:.3f} sec".format(time.time() - self.t_start))
        
        import traceback

        errors = [e for e in self.elements.values() if isinstance(e, Exception)]

        parser_errors = [e for e in errors if isinstance(e, ParserError)]
        unhandled_errors = [e for e in errors if e not in parser_errors]
        
        error_lines = [str(e.line) for e in errors if isinstance(e, ParserError)]
        from rich import inspect
        context = {
            "traceback": traceback,
            "errors": errors,
            "parser_errors": parser_errors,
            "unhandled_errors": unhandled_errors,
            "error_lines": error_lines, 
            "alt": altair, 
            "inspect": inspect}
        if 'model' in self.elements:
            model = self.elements['model']
            from math import nan
            steady_values = str({v: model.context['steady_states'].get(v,nan) for v in model.symbols['variables']}) 
            context.update({'steady_values':steady_values})

        if 'dr' in self.elements:
            dr = self.elements['dr']
            evs = abs(dr.evs)
            n = len(evs)//2
            if evs[n-1] < 1 < evs[n]:
                bk_check = True
            else:
                bk_check = False
            context.update({"bk_check": bk_check,"jacs": dr.coefficients_as_df()})

        d = {k: w for k,w in self.elements.items() if not isinstance(k,int)} | context

        txt = template.substitute(**d)

        for e in unhandled_errors:
            raise(e)
            # print(e)
        return txt

    def __call__(self, *s, **options):

        for e in s:
            self.elements[len(self.elements)+1] = e

        for k,w in options.items():
            self.elements[k] = w


def dsge_report(txt: str = None, filename: str = None, **options) -> Report:

    d = {}

    output_type = options.get("output_type", "markown")
    check_output = options.get("check_output", False)

    report = Report(output_type=output_type)

    if check_output:
        try:
            exec(txt,d,d)
        except Exception as e:
            report(str(e))
            return report
        try:
            return d["html"]
        except Exception as e:
            report(str(e))
            return report


    import time

    report(code = txt)

    try:
        if txt is not None:
            if filename is None:
                filename = "unknown"
        elif filename is not None:
            txt = open(filename).read()
        else:
            raise ValueError("Either `txt` or `filename` must be provided.")

        if filename.endswith(".mod"):
            preprocessor = options.get("modfile-preprocessor", "dynare")
            if preprocessor == "dynare":
                from dyno.modfile import DynareModel as DynoModel
            else:
                from dyno.dynofile import LDynoModel as DynoModel
            model = DynoModel(filename=filename, txt=txt) # =txt, filename=filename)
        elif filename.endswith(".dyno.yaml"):
            from dyno.yamlfile import YAMLFile
            model = YAMLFile(txt=txt)
        elif filename.endswith(".dyno"):
            from dyno.dynofile import LDynoModel
            model = LDynoModel(filename=filename, txt=txt)
        else:
            raise ValueError("Unsupported Model type")
        report(model=model)
    
        r = model.residuals

        report(residuals=r)
        if model.checks['deterministic']:
            from dyno.solver import deterministic_solve
            sim = deterministic_solve(model)
            report(sim={'Perfect Foresight': sim})
        else:
            dr = model.solve()
            report(dr=dr)
            sim = dr.irfs()
            report(sim=sim)

        from dyno.plots import plot_irfs
        fig = plot_irfs(sim)
        report(fig=fig)

    except Exception as e:
        report(e)
        return report

    from IPython.display import display, HTML, Markdown
    display(Markdown(report._repr_markdown_()))
    display(fig)
    display(Markdown("---"))