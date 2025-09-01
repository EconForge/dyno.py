files = [
    "example1.mod",
    "example2.mod",
    "NK_baseline.mod",
    "example3.mod",
    "Gali_2015.mod",
]

# TODO: for excluded file, check the error is meaningful
exclude = ["NK_baseline.mod"]  # uses an external steady_state file

unsupported = [
    "example1.mod",  # Uses native statement
    "example3.mod",  # calls steady_state function
    "Gali_2015.mod",  # calls external funciton in steady-state
]

from dyno import modfile
import pytest

files = [f for f in files if not (f in exclude)]


@pytest.mark.parametrize("filename", files)
def test_modfile_import(filename):

    f = filename

    filename = "examples/modfiles/" + f

    try:

        mod = modfile.Modfile(filename)
        sol = mod.compute()
        print(sol)

        assert True

    except Exception as e:
        assert f in unsupported

        assert isinstance(e, modfile.UnsupportedFeatureException)
