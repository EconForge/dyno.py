files = [
    "neo.yaml",
]

# TODO: for excluded file, check the error is meaningful
exclude = []

from dyno import YAMLFile
import pytest


# @pytest.mark.nondestructive
@pytest.mark.parametrize("filename", files)
def test_yamlfile_import(filename):

    f = filename
    print(f"Importing {f}")
    filename = "examples/" + f

    model = YAMLFile(filename)

    assert True
