def test_dynofile_modfile_equivalence():

    print("TEST dynofile/modfile equivalence")

    from dyno.symbolic_model import SymbolicModel as Model
    from dyno.modfile import DynareModel

    model1 = DynoModel("examples/RBC.dyno")
    model2 = DynareModel("examples/modfiles/RBC.mod")

    # d1 = (model1.get_calibration())
    # d2 = (model2.calibration)

    # d = ( {k: (d1[k], d2[k], d1[k]-d2[k]) for k in d1.keys() | d2.keys() if k in d1 and k in d2 } )

    # res1 = model1.compute()
    # res2 = model2.compute()

    dr1 = model1.solve()
    dr2 = model2.solve()
    vars1 = model1.symbols["endogenous"]
    vars2 = model2.symbols["endogenous"]
    import numpy as np

    XX = np.concatenate([dr1.X[:, vars1.index(sym)][:, None] for sym in vars2], axis=1)
    XX = np.concatenate([XX[vars1.index(sym), :][None, :] for sym in vars2], axis=0)

    # variable ordering is different, we need to reorder
    vare1 = model1.symbols["exogenous"]
    vare2 = model2.symbols["exogenous"]
    import numpy as np

    YY = np.concatenate([dr1.Y[:, vare1.index(sym)][:, None] for sym in vare2], axis=1)
    YY = np.concatenate([YY[vars1.index(sym), :][None, :] for sym in vars2], axis=0)

    assert abs(XX - dr2.X).max() < 1e-10
    assert abs(YY - dr2.Y).max() < 1e-10

    # It works !

    print(model1.symbols["endogenous"])
    print(model2.symbols["endogenous"])


def test_modfile_lark_preprocessor_equivalence():

    print("TEST dynofile/modfile equivalence")

    from dyno.symbolic_model import SymbolicModel as Model
    from dyno.modfile import DynareModel

    model1 = DynoModel("examples/modfiles/RBC.mod")
    model2 = DynareModel("examples/modfiles/RBC.mod")

    # d1 = (model1.calibration)
    # d2 = (model2.calibration)

    # d = ( {k: (d1[k], d2[k], d1[k]-d2[k]) for k in d1.keys() | d2.keys() if k in d1 and k in d2 } )

    res1 = model1.residuals
    res2 = model2.residuals

    print("Check equations")
    for eq1, eq2 in zip(model1.equations, model2.equations):
        print("Eq1: ", eq1)
        print("Eq2: ", eq2)
        # assert eq1 == eq2

    from rich.columns import Columns
    import numpy as np

    np.set_printoptions(precision=3)

    for i in range(len(res1)):
        print("Checking residuals and jacobians ", i, abs(res1[i] - res2[i]).max())
        assert (
            abs(res1[i] - res2[i]).max() < 1e-6
        )  # TODO: change for 1e-10 when symbolic calculations are restored

    # assert( len([k[2] for k in d.values() if abs(k[2]) > 1e-10]) ==0 )

    dr1 = model1.solve()
    dr2 = model2.solve()
    vars1 = model1.symbols["endogenous"]
    vars2 = model2.symbols["endogenous"]
    import numpy as np

    XX = np.concatenate([dr1.X[:, vars1.index(sym)][:, None] for sym in vars2], axis=1)
    XX = np.concatenate([XX[vars1.index(sym), :][None, :] for sym in vars2], axis=0)

    assert abs(XX - dr2.X).max() < 1e-6
    # assert abs(dr2.Y - dr2.Y).max() < 1e-6

    # TODO: check that dr1.Y and dr2.Y are identical modulo some reordering
