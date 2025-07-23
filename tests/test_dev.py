from dyno import modfile
import pytest

def test_modfile_describe():
    files = [
        # "example1.mod",
        # "example2.mod",
        # "example3.mod",
        # "Gali_2015.mod",
        "NK_baseline.mod"
    ]

    # TODO: for excluded file, check the error is meaningful
    exclude = []

    f = files[0]

    filename = "examples/modfiles/" + f

    model = modfile.Modfile(filename)

    print(model.describe())


# print(model.solve())
