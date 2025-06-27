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
    "example3.mod",  # calls steady_state function
    "Gali_2015.mod",  # calls external funciton in steady-state
]

from dyno import modfile_preprocessor as preprocessor
from dyno import modfile as lark
import pytest

files = [f for f in files if not (f in exclude)]


@pytest.fixture(params=[preprocessor, lark])
def modfile(request):
    return request.param


# @pytest.mark.nondestructive
@pytest.mark.parametrize("filename", files)
def test_modfile_import(filename, modfile):

    f = filename

    filename = "examples/modfiles/" + f

    try:

        mod = modfile.Modfile(filename)
        sol = mod.compute()
        print(sol)

        assert True

    except Exception as e:

        from dyno.model import UnsupportedDynareFeature

        assert f in unsupported

        assert isinstance(e, UnsupportedDynareFeature)
