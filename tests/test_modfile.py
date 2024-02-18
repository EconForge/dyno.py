files = [
    "example1.mod",
    "example2.mod",
    "example3.mod",
    "Gali_2015.mod",
    "NK_baseline.mod"
]

# TODO: for excluded file, check the error is meaningful
exclude = []

from dyno.modfile import Modfile

import pytest


# @pytest.mark.nondestructive
@pytest.mark.parametrize("filename",files)
def test_modfile_import(filename):

    f = filename
    print(f"Importing {f}")
    filename = "examples/modfiles/" + f
    mod = Modfile(filename)
    print(mod.calibration)
    print(mod.exogenous)
    assert True