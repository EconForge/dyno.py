class Report:

    def __init__(self, *elements):

        self.elements = elements

    def _repr_html_(self):

        reprs = []
        for r in self.elements:
            if hasattr(r, "_repr_html_"):
                reprs.append(r._repr_html_())
            else:
                reprs.append(f"<pre>{str(r)}</pre>")
        
        html = "<br>".join([str(e) for e in reprs])
        
        return html

def dsge_report(txt: str=None, filename: str=None, **options) -> Report:

    import time

    t1 = time.time()
    elements = []
    try:
        if txt is not None:
            if filename is None:
                filename = "unknown"
        elif filename is not None:
            txt = open(filename).read()
        else:
            raise ValueError("Either `txt` or `filename` must be provided.")
    except Exception as e:
        elements.append(e)
        return Report(*elements)

    try:
        if filename.endswith(".mod"):
            preprocessor = options.get("modfile_preprocessor", 'dynare')
            if preprocessor == "dynare":
                from dyno.modfile import DynareModel
            else:
                from dyno.modfile_lark import DynareModel
            model = DynareModel(txt=txt)
        elif filename.endswith(".dyno.yaml"):
            from dyno.yamlfile import YAMLFile
            model = YAMLFile(txt=txt)
        else:
            raise ValueError("Unsupported Model type")
            
        elements.append(model)
    except Exception as e:
        elements.append(e)
        return Report(*elements)

    try:
        r = model.compute()
        err = abs(r).max()
        # elements.append(r)
        if err > 1e-6:
            raise Exception(f"Steady-State Error\n. Residuals: {abs(r)}")
    except Exception as e:
        elements.append(e)
        return Report(*elements)

    try:
        dr = model.solve()
        elements.append(dr)
    except Exception as e:
        elements.append(e)
        return Report(*elements)

    t2 = time.time()

    elements.append(f"Time: {t2 - t1:.2f} sec")
    
    return Report(*elements)