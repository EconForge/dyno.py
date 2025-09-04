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

def dsge_report(txt: str=None, filename: str=None) -> Report:

    from dyno.modfile import DynareModel

    elements = []

    try:
        if filename is not None:
            model = DynareModel(filename=filename)
        elif txt is not None:
            model = DynareModel(txt=txt)
        else:
            raise ValueError("Either `txt` or `filename` must be provided.")
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

    return Report(*elements)