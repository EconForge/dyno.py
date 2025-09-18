from IPython.display import display, HTML, Markdown, TextDisplayObject
import time
import rich

class Report:

    def __init__(self, *elements, **options):

        self.elements = {}
        self.output_type = options.get("output_type", "markdown")
        self.t_start = time.time()


    # def __repr__(self):

    #     self("Elapsed time: {:.3f} sec".format(time.time() - self.t_start))

    #     if self.output_type != "text":
    #         return None
        

    def _repr_html_(self):
        import time
        self("Elapsed time: {:.3f} sec".format(time.time() - self.t_start))


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

        if self.output_type != "markdown":
            return None

        code = f"""
# Model: {self.elements.get("name")}



::: {{note}} Equations
:class: dropdown

{str.join("\n\n", self.elements['model'].equations)})

:::

Here's some *text*

1. a list

> a quote

{{emphasis}}`content`


::: {{tip}} Code
:class: dropdown
```{{code-block}} python
:linenos:
:emphasize-lines: 2,3

{self.elements['code']}
```
:::

![](clippy.png)


# Decision Rule

::: {{tip}} Eigenvalues

{self.elements['dr'].evs}

:::

# Simulation

---
""" + str.join("\n", [str(e) for e in self.elements.values()])

        return code

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

    try:
        dr = model.solve()
        report(dr=dr)
    except Exception as e:
        report(e)
        return report


    return report