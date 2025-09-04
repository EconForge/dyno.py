def test_solve_KR2000_STAT():

    ############################################################################
    # Test KR2000_STAT.mod
    ############################################################################

    from dyno.modfile import DynareModel

    model = DynareModel("examples/modfiles/model_KR2000_STAT.mod")
    dr = model.solve()

    from dyno.solver import moments
    S, S_e = moments(dr.X, dr.Y, dr.Σ)
