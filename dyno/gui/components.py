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
def SolutionViewer(dr_):

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

    df_moments = pd.DataFrame(
        hh_y,
        columns=["{}[t]".format(e) for e in (dr.symbols["endogenous"])],
        index=["{}[t]".format(e) for e in (dr.symbols["endogenous"])],
    )

    solara.Markdown("__Moments__")

    solara.display(df_moments)


# def one_plot(irfs_, c, selects):

#     irfs = irfs_.value

#     k0 = [*irfs.keys()][0]

#     x_ = irfs[k0][c].index


#     # create scales
#     xs = bqplot.LinearScale(min=0,max=len(x_)+1)
#     # ys = bqplot.LinearScale()

#     colors=["blue","red"]
#     # with iw.Card(outlinedd=True,_style="width: 350px; height: 250px"):
#     lines = [
#         bqplot.Lines(x=x_, y=irfs[k][c], scales={"x": xs, "y": bqplot.LinearScale()}, labels=[k], colors=colors[i])
#         for (i,k) in enumerate(irfs['shocks'].unique()) if k in selects.value
#     ]
#     # create axes objects
#     # xax = bqplot.Axis(scale=xs, grid_lines="solid", label='t')
#     # yax = bqplot.Axis(scale=ys, orientation="vertical", label=c, grid_lines="solid")

#     # create the figure object (renders in the output cell)
#     return bqplot.Figure(marks=lines, legend=True, transition=True)


# from reacton import ipyvuetify as iw


# @solara.component
# def SimulViewer(irfs, sim_grid, selects):

#     cols = [str(e) for e in [*irfs.value.values()][0].columns[1:]]
#     n = len(cols)

#     ind, set_ind = solara.use_state(cols[0])


#     if not sim_grid.value:
#         with iw.Window(v_model=ind, on_v_model=set_ind, show_arrows=True):
#             for (i,c) in enumerate(cols):
#                 with iw.WindowItem(value=c):
#                     # with iw.Card():
#                         one_plot(irfs, c, selects)
#     else:
#         with solara.ColumnsResponsive():
#             for (i,c) in enumerate(cols):
#                     # with iw.Card():
#                         one_plot(irfs, c, selects)


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
