import dyno
from dyno import YAMLFile


def test_solution_ti():

    model = YAMLFile("examples/neo.yaml")

    sol = model.solve(method="ti")

    sim = dyno.simulate(sol)


def test_solution_qz():

    from dyno import YAMLFile

    model = YAMLFile("examples/neo.yaml")

    sol = model.solve(method="qz")

    sim = dyno.simulate(sol)
