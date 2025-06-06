import solara

import solara
from solara.alias import rv
import pandas as pd
import numpy as np

# from reacton import bqplot
import numpy as np

# import dyno
import time


@solara.component
def SolutionViewer(dr_, moments_, use_hpfilter_):

    from IPython.core.display import display_html

    dr = dr_.value

    solara.Markdown("__Eigenvalues__")

    evv = pd.DataFrame(
        [np.abs(dr.evs)], columns=[i + 1 for i in range(len(dr.evs))], index=["λ"]
    )

    solara.display(evv)
    # html_evs = display_html(evv)

    # print(html_evs)
    # solara.HTML(tag="table", unsafe_innerHTML=html_evs, style="font-family: monospace")

    solara.Markdown("__Steady-state__")

    ss = pd.DataFrame(
        [dr.x0], columns=["{}".format(e) for e in dr.symbols["endogenous"]]
    )

    solara.display(ss)

    # solara.Markdown("### Decision Rule")
    solara.Markdown("__Decision Rule__")

    hh_y = dr.X
    hh_e = dr.Y

    df = pd.DataFrame(
        np.concatenate([hh_y, hh_e], axis=1),
        columns=["{}[t-1]".format(e) for e in dr.symbols["endogenous"]]
        + ["{}[t]".format(e) for e in (dr.symbols["exogenous"])],
    )

    df.index = ["{}[t]".format(e) for e in dr.symbols["endogenous"]]

    solara.display(df)

    if moments_.value:

        from dyno.solver import moments

        Σ0, Σ = moments(dr.X, dr.Y, dr.Σ)

        df_cmoments = pd.DataFrame(
            Σ0,
            columns=["{}[t]".format(e) for e in (dr.symbols["endogenous"])],
            index=["{}[t]".format(e) for e in (dr.symbols["endogenous"])],
        )

        df_umoments = pd.DataFrame(
            Σ,
            columns=["{}[t]".format(e) for e in (dr.symbols["endogenous"])],
            index=["{}[t]".format(e) for e in (dr.symbols["endogenous"])],
        )

        solara.Markdown("__Unconditional Moments__")
        solara.display(df_umoments)

        solara.Markdown("__Conditional Moments__")
        solara.display(df_cmoments)


def SimulViewer2(irfs_, sim_grid, selects):

    import plotly.express as px

    irfs = irfs_.value

    fig = px.line(
        irfs, x="t", y="value", color="shock", facet_col="variable", facet_col_wrap=2
    )
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    fig.update_yaxes(title_text="", matches=None)
    fig.update_xaxes(title_text="")

    solara.FigurePlotly(fig)

    solara.Markdown("The values on the y axes are %-deviations from the steady state.")


from reacton import ipyvuetify as iw


@solara.component
def ParameterChooser(args, parameters, import_model, text):

    with iw.Container() as main:
        for k, val in parameters.items():
            solara.SliderFloat(
                k,
                value=val,
                min=args[k][1],
                max=args[k][2],
                on_value=lambda v: import_model(text.value),
                step=(args[k][2] - args[k][1]) / 10,
            )

    return main
