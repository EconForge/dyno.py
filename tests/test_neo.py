import dyno
from dyno import YAMLFile


def test_solution_ti():


    # from dyno import YAMLFile
    # model = YAMLFile("examples/neo.yaml")
     
    from dyno.dynofile import LDynoModel
    model = LDynoModel("examples/neo.dyno")

    model.describe()

    sol = model.solve(method="ti")

    sim = dyno.simulate(sol)


def test_solution_qz():

    # from dyno import YAMLFile
    # model = YAMLFile("examples/neo.yaml")

    from dyno.dynofile import LDynoModel

    model = LDynoModel("examples/neo.dyno")


    sol = model.solve(method="qz")

    sim = dyno.simulate(sol)


if __name__ == "__main__":
    
    test_solution_ti()
    test_solution_qz()