import dyno


def test_solution_ti():
    from dyno.dyno_model import DynoModel

    model = DynoModel("examples/neo.dyno")

    model.describe()

    sol = model.solve(method="ti")

    sim = dyno.simulate(sol)


def test_solution_qz():
    from dyno.dyno_model import DynoModel

    model = DynoModel("examples/neo.dyno")

    sol = model.solve(method="qz")

    sim = dyno.simulate(sol)


if __name__ == "__main__":

    test_solution_ti()
    test_solution_qz()
