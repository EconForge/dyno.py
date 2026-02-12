def test_modfile_describe():
    files = [
        "example1.mod",
        "example2.mod",
        # "example3.mod", # has steady-state external file
        # "Gali_2015.mod", # has steady-state() function
        "NK_baseline.mod",
    ]

    # TODO: for excluded file, check the error is meaningful
    exclude = []

    for f in files:
        print("Trying to import and describe", f)
        filename = "examples/modfiles/" + f
        from dyno.symbolic_model import DynoModel

        model = DynoModel(filename)
        print(model.describe())


if __name__ == "__main__":
    test_modfile_describe()
# print(model.solve())
