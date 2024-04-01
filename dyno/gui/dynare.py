def dyno_gui(filename):
    
    import solara
    from solara.alias import rv
    import pandas as pd
    import numpy as np
    # from reacton import bqplot 
    import numpy as np
    import dyno
    import time
    
    def sim_to_nsim(irfs):
    
        pdf = pd.concat(irfs).reset_index()
        ppdf = pdf.rename(columns={'level_0': 'shock','level_1': 't'})
    
        ppdf = ppdf.melt(id_vars=['shock','t'])
    
        return ppdf
    
    
    txt = open(filename).read()
    
    from dyno.modfile import Modfile
    model = Modfile(txt=txt)
    dr0 = model.solve()
    
    sim = sim_to_nsim(dyno.model.irfs(model, dr0))
    
    modfile = solara.reactive(txt)
    simul = solara.reactive(sim)
    
    sel = list(sim['shock'].unique())
    
    selects = solara.reactive( sel )
    
    
    dr = solara.reactive(dr0)
    ok_import = solara.reactive(True)
    ok_ss = solara.reactive(True)
    ok_bk = solara.reactive(True)
    ok = solara.reactive(True)
    
    use_qz = solara.reactive(True)
    
    
    model_description = solara.reactive(None)
    msg = solara.reactive("No problem")
    
    text = solara.reactive(txt)
    
    solution_time = solara.reactive((0.0,0.0, 0.0))
    sim_grid = solara.reactive(False)
    
    split_screen = solara.reactive(False)
    
    
    def import_model(txt):
    
        text.value = txt
    
        model = None
        # msg = solara.reactive("No problem")
        
        try:
    
            ta = time.time()
            model = Modfile(txt=txt)
            tb = time.time()
            # model_description.value = model.describe()
            ok_import.value = True
        except Exception as e:
            ok_import.value = False
            ok_ss.value = False
            ok_bk.value = False
            msg.value = str(e)
            ok.value = ok_import.value & ok_ss.value & ok_bk.value
    
            return
    
        try:
            tc = time.time()
            r = model.compute()
            td = time.time()
            err = abs(r).max()
            if err>1e-6:
                raise Exception(f"Steady-State Error\n. Residuals: {abs(r)}")
                
            ok_ss.value = True
        except Exception as e:
            msg.value = f"Oups! {e}"
            ok_ss.value = False
            ok_bk.value = False
            ok.value = ok_import.value & ok_ss.value & ok_bk.value
    
            return
        
        try:
    
            t1 = time.time()
    
            if use_qz:
                method = "qz"
            else:
                method = "ti"
    
            dr.value = model.solve(method=method)
    
            t2 = time.time()
    
            solution_time.value = (tb-ta,td-tc,t2-t1)
            
            ok_bk.value = True
    
        except Exception as e:
    
            msg.value = f"Oups! {e}"
            ok_bk.value = False
            ok.value = ok_import.value & ok_ss.value & ok_bk.value
    
    
        if ok_bk.value:
            sim = dyno.model.irfs(model, dr.value)
    
    
            simul.value = sim_to_nsim(sim)
    
    
        # ok.value = ok_import.value & ok_ss.value & ok_bk.value
        
    
    import copy
    
    @solara.component
    def ModelEditor():
                            
        with solara.Column():
            
            rv.Textarea(v_model=text.value, on_v_model=import_model, rows=20, style_="font-family: monospace")
    
    @solara.component
    def SolutionViewer(dr_):
    
        from IPython.core.display import display_html
    
        dr = dr_.value
    
        solara.Markdown("__Eigenvalues__")
    
        evv = pd.DataFrame(
            [np.abs(dr.evs)],
            columns = [i+1 for i in range(len(dr.evs))],
            index=["λ"]
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
            columns=["{}[t-1]".format(e) for e in dr.symbols['endogenous']]+["{}[t]".format(e) for e in (dr.symbols['exogenous'])]
        )
    
        df.index = ["{}[t]".format(e) for e in dr.symbols['endogenous']]
    
        solara.display(df)
    

        df_moments = pd.DataFrame(
            hh_y,
            columns=["{}[t]".format(e) for e in (dr.symbols['endogenous'])],
            index=["{}[t]".format(e) for e in (dr.symbols['endogenous'])]
        )

        solara.Markdown("__Moments__")

        solara.display(df_moments)


    
    
    def one_plot(irfs_, c, selects):
    
        irfs = irfs_.value
    
        k0 = [*irfs.keys()][0]
    
        x_ = irfs[k0][c].index
    
    
        # create scales
        xs = bqplot.LinearScale(min=0,max=len(x_)+1)
        # ys = bqplot.LinearScale()
    
        colors=["blue","red"]
        # with iw.Card(outlinedd=True,_style="width: 350px; height: 250px"):
        lines = [
            bqplot.Lines(x=x_, y=irfs[k][c], scales={"x": xs, "y": bqplot.LinearScale()}, labels=[k], colors=colors[i])
            for (i,k) in enumerate(irfs['shocks'].unique()) if k in selects.value
        ]
        # create axes objects
        # xax = bqplot.Axis(scale=xs, grid_lines="solid", label='t')
        # yax = bqplot.Axis(scale=ys, orientation="vertical", label=c, grid_lines="solid")
    
        # create the figure object (renders in the output cell)
        return bqplot.Figure(marks=lines, legend=True, transition=True)
    
    
    from reacton import ipyvuetify as iw
    
    
    
    @solara.component
    def SimulViewer(irfs, sim_grid):
        
        cols = [str(e) for e in [*irfs.value.values()][0].columns[1:]]
        n = len(cols)
    
        ind, set_ind = solara.use_state(cols[0])
    
    
    
        if not sim_grid.value:
            with iw.Window(v_model=ind, on_v_model=set_ind, show_arrows=True):
                for (i,c) in enumerate(cols):
                    with iw.WindowItem(value=c):
                        # with iw.Card():
                            one_plot(irfs, c, selects)
        else:
            with solara.ColumnsResponsive():
                for (i,c) in enumerate(cols):
                        # with iw.Card():
                            one_plot(irfs, c, selects)
            
    
    
    def SimulViewer2(irfs_, sim_grid):
            
        import plotly.express as px
        
        irfs = irfs_.value
    
        fig = px.line(irfs, x='t', y='value', color='shock', facet_col="variable", facet_col_wrap=2)
        fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
        fig.update_yaxes(title_text = "", matches=None)
        fig.update_xaxes(title_text = "")
    
        solara.FigurePlotly(fig)
    
    
    @solara.component
    def Diagnostic():
    
            with solara.Card("Diagnostic"):
                if model_description.value is not None:
                    solara.Text(model_description.value)
                if not ok_bk.value:
                    solara.Warning(label=msg.value)
                else:
                    v = solution_time.value
                    solara.Success(label=f"All Good!")
                                   
                solara.Checkbox(label="Import", value=ok_import)
                solara.Checkbox(label="Steady-state", value=ok_ss)
                solara.Checkbox(label="Blanchard-Kahn", value=ok_bk)
    
    
    @solara.component
    def Page():
        
        with solara.Head():
            solara.Title("Dyno: {}".format(filename))
        with solara.Sidebar():
            with solara.Column():
                
                Diagnostic()
    
    
                # with solara.Card("Simul Options"):
                #     solara.Checkbox(label="Use Qz", value=use_qz)
                #     solara.Checkbox(label="Grid", value=sim_grid)
                #     # solara.Checkbox(label="Split", value=split_screen)
    
                #     solara.SelectMultiple(
                #         label='Shocks',
                #         all_values=sel,
                #         values=selects,
                #         on_value=lambda w: selects.set(w),
                #     )
    
        if not split_screen.value:
            with solara.lab.Tabs():
                with solara.lab.Tab("Model"):
                    with solara.Card(elevation=2):
                        ModelEditor()
                with solara.lab.Tab("Solution"):
                    with solara.Card(elevation=2):
                        SolutionViewer(dr)
    
                with solara.lab.Tab("Simulation"):
                    with solara.Card(elevation=2):
                        SimulViewer2(simul, sim_grid)
                    
                    # solara.DataFrame(simul.value)
    
    return Page()