from IPython.display import display, HTML, Markdown, TextDisplayObject
import time
import rich
import pandas
import altair
import tempita

import altair as alt
# load a simple dataset as a pandas DataFrame
from vega_datasets import data

cars = data.cars()
ch = alt.Chart(cars).mark_point().encode(
    x='Horsepower',
    y='Miles_per_Gallon',
    color='Origin',
)


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
                

{[if model is not None]}
                            
## Model Summary

- *name*:  {[model.name]}
- *variables*:  {[str.join(", ", model.variables)]}
- *parameters*:  {[str.join(", ", model.parameters)]}

:::{dropdown} Calibration
Parameter values
```{code} python
{[ str(model.calibration) ]}
```
Steady state values
```{code} python
{[ str(model.steady_state()) ]}
```
:::
                            
{[if abs(residuals).max() > 1e-6]}
:::{warning} Non zero residuals
:class: dropdown
The model has non zero residuals after calibration. This may be due to a missing steady-state
calculation or an error in the model equations.
```{code} python
{[ str(residuals) ]}
```
:::
{[endif]}

---
{[endif]}


{[if dr is not None]}
                    
## Solution
                            
First order approximation.
                            
Ressiduals: {[residuals]}
                                
System Eigenvalues: {[dr.evs.real]}

Blanchard-Kahn conditions: {[ "satisfied" if bk_check else "not satisfied" ]}

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
---                   
{[endif]}
                            

{[if sim is not None]}

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
                            
                            
{[for er in errors]}
```
{[str(er)]}
```
{[endfor]}
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
    
        errors = [w for w in self.elements.values() if isinstance(w, Exception)]
        context = {"errors": errors, "alt": altair}
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

        if 'fig' in self.elements:
            display(self.elements['fig'])

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
        except Exception:
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
    except Exception as e:
        report(e)
        return report

    try:
        if filename.endswith(".mod"):
            preprocessor = options.get("modfile_preprocessor", "dynare")
            if preprocessor == "dynare":
                from dyno.modfile import DynareModel
            else:
                from dyno.modfile_lark import DynareModel
            model = DynareModel(txt=txt)
        elif filename.endswith(".dyno.yaml"):
            from dyno.yamlfile import YAMLFile
            model = YAMLFile(txt=txt)
        elif filename.endswith(".dyno"):
            from dyno.dynofile import DynoModel
            model = DynoModel(txt=txt)
        else:
            raise ValueError("Unsupported Model type")

        report(model=model)
    except Exception as e:
        report(e)
        return report

    try:
        r = model.compute()
        err = abs(r).max()
        report(residuals=r)
        if err > 1e-6:
            raise Exception(f"Steady-State Error\n. Residuals: {abs(r)}")
    except Exception as e:
        report(e)
        return report

    if model.checks['deterministic']:
        print("Deterministic model")
        try:
            from dyno.solver import deterministic_solve
            sim = deterministic_solve(model)
            report(sim={'Perfect Foresight': sim})
        except Exception as e:
            report(e)
            return report
        
        try:
            sim.plot()
        except Exception as e:
            report(e)
            return report
    else:
        print("Stochastic model")

        try:
            dr = model.solve()
            report(dr=dr)
        except Exception as e:
            report(e)
            return report
        
        try:
            sim = dr.irfs()
            report(sim=sim)
        except Exception as e:
            report(e)
            return report

        try:
            fig = dr.plot()
            report(fig=fig)
        except Exception as e:
            report(e)
            return report

    return report