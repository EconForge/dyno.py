def test_solution_ti():

    import dyno

    model = dyno.import_file("examples/neo.yaml")

    sol = model.solve(method="ti")

    sim = dyno.simulate(sol)


def test_solution_qz():

    import dyno

    model = dyno.import_file("examples/neo.yaml")

    sol = model.solve(method="qz")

    sim = dyno.simulate(sol)
