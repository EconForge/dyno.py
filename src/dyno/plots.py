from dyno.simul import sim_to_nsim


def plot_irfs(sim):
    if isinstance(sim, dict):
        return plot_irfs_altair(sim)
    else:
        return plot_irf_altair(sim)


def plot_irf_altair(sim):

    import altair as alt

    sim = sim.melt(id_vars=["t"])

    ch = (
        alt.Chart(sim)
        .mark_line()
        .encode(x="t", y="value", facet=alt.Facet("variable", columns=2))
        .properties(width=200, height=100)
        .resolve_scale(y="independent")
        .interactive()
    )
    return ch


def plot_irfs_altair(sim):

    import altair as alt

    sim = sim_to_nsim(sim)

    ch = (
        alt.Chart(sim)
        .mark_line()
        .encode(x="t", y="value", color="shock", facet=alt.Facet("variable", columns=2))
        .properties(width=200, height=100)
        .resolve_scale(y="independent")
        .interactive()
    )
    return ch


def plot_irfs_plotly(sim):

    import plotly.express as px

    plots = sim_to_nsim(sim)

    fig = px.line(
        plots,
        x="t",
        y="value",
        color="shock",
        facet_col="variable",
        facet_col_wrap=2,
    )

    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    fig.update_yaxes(title_text="", matches=None)
    fig.update_xaxes(title_text="")

    return fig


def plot_irf_plotly(sim):

    sim["t"] = sim.index
    plots = sim.melt(id_vars=["t"])

    fig = px.line(
        plots,
        x="t",
        y="value",
        facet_col="variable",
        facet_col_wrap=2,
    )

    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    fig.update_yaxes(title_text="", matches=None)
    fig.update_xaxes(title_text="")

    return fig
