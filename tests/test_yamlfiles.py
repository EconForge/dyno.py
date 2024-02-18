files = [
    "neo.yaml",
]

# TODO: for excluded file, check the error is meaningful
exclude = []

import dyno
import pytest


# @pytest.mark.nondestructive
@pytest.mark.parametrize("filename",files)
def test_yamlfile_import(filename):

    f = filename
    print(f"Importing {f}")
    filename = "examples/" + f

    model = dyno.import_file(filename)
    
    assert True