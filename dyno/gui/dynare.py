def dyno_gui(filename, parchoice={}):

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
        ppdf = pdf.rename(columns={"level_0": "shock", "level_1": "t"})

        ppdf = ppdf.melt(id_vars=["shock", "t"])

        return ppdf

    txt = open(filename).read()

    from dyno.modfile import Modfile

    model = Modfile(txt=txt)
    dr0 = model.solve()

    sim = sim_to_nsim(dyno.model.irfs(model, dr0))

    modfile = solara.reactive(txt)
    simul = solara.reactive(sim)

    sel = list(sim["shock"].unique())

    selects = solara.reactive(sel)

    dr = solara.reactive(dr0)
    ok_import = solara.reactive(True)
    ok_ss = solara.reactive(True)
    ok_bk = solara.reactive(True)
    ok = solara.reactive(True)

    use_qz = solara.reactive(True)

    model_description = solara.reactive(None)
    msg = solara.reactive("No problem")

    text = solara.reactive(txt)

    solution_time = solara.reactive((0.0, 0.0, 0.0))
    sim_grid = solara.reactive(False)

    split_screen = solara.reactive(False)

    import copy

    args_0 = copy.copy(parchoice)

    parameters = {k: solara.reactive(v[0]) for (k, v) in args_0.items()}

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
            if err > 1e-6:
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

            parms = {k: v for k, v in parameters.items()}

            dr.value = model.solve(method=method, calibration=parms)

            t2 = time.time()

            solution_time.value = (tb - ta, td - tc, t2 - t1)

            ok_bk.value = True

        except Exception as e:

            msg.value = f"Oups! {e}"
            ok_bk.value = False
            ok.value = ok_import.value & ok_ss.value & ok_bk.value

        if ok_bk.value:
            sim = dyno.model.irfs(model, dr.value)

            simul.value = sim_to_nsim(sim)

        # ok.value = ok_import.value & ok_ss.value & ok_bk.value

    @solara.component
    def ModelEditor():

        with solara.Column():

            rv.Textarea(
                v_model=text.value,
                on_v_model=import_model,
                rows=20,
                style_="font-family: monospace",
            )

    from .components import SolutionViewer, SimulViewer2
    from .components import ParameterChooser

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

                with solara.Card("Parameters"):
                    ParameterChooser(args_0, parameters, import_model, text)

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
                        SimulViewer2(simul, sim_grid, selects)

                    # solara.DataFrame(simul.value)

    return Page()
