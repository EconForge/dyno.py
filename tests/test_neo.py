def test_all():

    import dyno

    model = dyno.import_file("examples/neo.yaml")

    sol = model.solve()

    sim = dyno.simulate(sol)

    