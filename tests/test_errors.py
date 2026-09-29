from dyno.errors import DynareParserError


class _DummyDynareError(Exception):
    pass


def test_dynare_parser_error_extracts_line_and_column():
    err = _DummyDynareError("syntax error, unexpected TIMES: line 46, col 34")

    parsed = DynareParserError(err)  # type: ignore[arg-type]

    assert parsed.line == 46
    assert parsed.column == 34


def test_dynare_parser_error_handles_missing_location():
    err = _DummyDynareError("unsupported feature: native statement")

    parsed = DynareParserError(err)  # type: ignore[arg-type]

    assert parsed.line is None
    assert parsed.column is None


def test_dynare_parser_error_extracts_cols_range():
    err = _DummyDynareError(
        "ERROR: in_memory.mod: line 3, cols 1-6: y is not a parameter"
    )

    parsed = DynareParserError(err)  # type: ignore[arg-type]

    assert parsed.line == 3
    assert parsed.column == 1


def test_dynare_parser_error_extracts_eof_cols():
    err = _DummyDynareError(
        "ERROR: in_memory.mod: line 4, cols 1-0: syntax error, unexpected end of file"
    )

    parsed = DynareParserError(err)  # type: ignore[arg-type]

    assert parsed.line == 4
    assert parsed.column == 1


def test_dynare_parser_error_extracts_lines_range():
    err = _DummyDynareError(
        "ERROR: in_memory.mod: lines 10-12, cols 5-8: unexpected token"
    )

    parsed = DynareParserError(err)  # type: ignore[arg-type]

    assert parsed.line == 10
    assert parsed.column == 5


def test_dynare_parser_error_extracts_line_without_col():
    err = _DummyDynareError("ERROR: in_memory.mod: line 15: some error message")

    parsed = DynareParserError(err)  # type: ignore[arg-type]

    assert parsed.line == 15
    assert parsed.column is None


def test_dynare_parser_error_reads_direct_attributes():
    class _ErrorWithAttrs(Exception):
        line = 7
        column = 12

    err = _ErrorWithAttrs("some message without location text")
    parsed = DynareParserError(err)

    assert parsed.line == 7
    assert parsed.column == 12


def test_dynare_parser_error_reads_begin_attrs_and_location():
    class _DummyLoc:
        line = 9
        column = 14

    class _ErrorWithLoc(Exception):
        location = _DummyLoc()

    err = _ErrorWithLoc("custom error")
    parsed = DynareParserError(err)

    assert parsed.line == 9
    assert parsed.column == 14


import pytest
import warnings
from dyno import DynoModel
from dyno.errors import (
    UndefinedSymbolWarning,
    UndefinedSymbolError,
    SystemStructureError,
)


def test_undefined_parameter_in_equations_emits_warning():
    txt = """
k[t] = alpha * k[t-1]
k[~] <- 1.0
"""
    with pytest.warns(UndefinedSymbolWarning, match="Undefined parameter.*alpha"):
        model = DynoModel(txt=txt)
    assert "alpha" in model.context["constants"]


def test_undefined_parameter_in_equations_strict_raises_error():
    txt = """
k[t] = alpha * k[t-1]
k[~] <- 1.0
"""
    with pytest.raises(UndefinedSymbolError, match="Undefined parameter.*alpha"):
        DynoModel(txt=txt, strict=True)


def test_check_raises_undefined_symbol_error_when_symbols_uninitialized():
    txt = """
k[t] = 0.5 * k[t-1]
"""
    model = DynoModel(txt=txt)
    with pytest.raises(UndefinedSymbolError, match="variables without steady state: k"):
        model.check()


def test_solve_raises_system_structure_error_when_system_not_square():
    txt = """
k[t] = 0.5 * k[t-1] + c[t]
k[~] <- 1.0
c[~] <- 0.5
"""
    model = DynoModel(txt=txt)
    with pytest.raises(
        SystemStructureError, match="Model has 1 equation.*2 endogenous variable"
    ):
        model.solve()


def test_unsupported_macroprocessor_raises_error():
    from dyno.errors import UnsupportedFeatureError

    mod_txt = """
    @#define ghh = 1
    var c;
    varexo e;
    parameters beta;
    beta = 0.99;
    model;
    c = beta * c(+1) + e;
    end;
    """
    # When preprocess=False, macro directives raise UnsupportedFeatureError in LModFile
    with pytest.raises(UnsupportedFeatureError) as exc_info:
        DynoModel(txt=mod_txt, filename="test.mod", preprocess=False)
    assert exc_info.value.feature == "macroprocessor"
    assert "macroprocessor" in str(exc_info.value)

    # When preprocess=True (default), it succeeds
    m = DynoModel(txt=mod_txt, filename="test.mod")
    assert "c" in m.symbols["variables"]


def test_unsupported_estimation_raises_error():
    from dyno.errors import UnsupportedFeatureError

    mod_txt = """
    var c;
    varexo e;
    parameters beta;
    beta = 0.99;
    model;
    c = beta * c(+1) + e;
    end;
    estimated_params;
    beta, normal_pdf, 0.99, 0.01;
    end;
    """
    with pytest.raises(UnsupportedFeatureError) as exc_info:
        DynoModel(txt=mod_txt, filename="test.mod")
    assert exc_info.value.feature == "estimation"
    assert "estimation" in str(exc_info.value)


def test_unsupported_occbin_raises_error():
    from dyno.errors import UnsupportedFeatureError

    mod_txt = """
    var c;
    varexo e;
    parameters beta;
    beta = 0.99;
    model;
    c = beta * c(+1) + e;
    end;
    occbin_constraints;
    name 'c_eq'; bind c < 0;
    end;
    """
    with pytest.raises(UnsupportedFeatureError) as exc_info:
        DynoModel(txt=mod_txt, filename="test.mod")
    assert exc_info.value.feature == "occbin"
    assert "Occbin" in str(exc_info.value)


def test_bkk_mod_unsupported_timing_error():
    from dyno.errors import SystemStructureError

    # strict=True raises on import
    with pytest.raises(SystemStructureError) as exc_info:
        DynoModel("examples/modfiles/bkk.mod", strict=True)
    msg = str(exc_info.value)
    assert "Unsupported timing" in msg
    assert "only -1, 0, 1 so far" in msg
    assert "K_H[t-4]" in msg

    # strict=False imports with warning, raises on run
    m_loose = DynoModel("examples/modfiles/bkk.mod", strict=False)
    with pytest.raises(SystemStructureError) as exc_run:
        m_loose.run()
    assert "Unsupported timing" in str(exc_run.value)


def test_model_local_variables_and_unary_plus():
    txt = """
    var c, k;
    varexo e;
    parameters alpha, beta;
    alpha = 0.33;
    beta = 0.99;
    model;
    # u = c^alpha;
    + c = + beta * u + e;
    k = alpha * k(-1);
    end;
    shocks;
    var e = 0.01;
    end;
    """
    model = DynoModel(txt=txt, filename="local_test.mod")
    assert len(model.equations) == 2


# ---------------------------------------------------------------------------
# Tests for newly-added UnsupportedFeatureError detections
# ---------------------------------------------------------------------------


def test_unsupported_deterministic_shocks_in_shocks_block():
    """periods/values inside a shocks block (perfect foresight syntax) must raise
    UnsupportedFeatureError, not a raw LARKParserError."""
    from dyno.errors import UnsupportedFeatureError

    mod_txt = """\
var c, k;
varexo z;
parameters beta, alpha, delta;
beta = 0.99;
alpha = 0.33;
delta = 0.025;
model;
c + k = alpha * k(-1)^alpha + (1-delta)*k(-1) + z;
c = beta * c(+1) * (alpha * k^(alpha-1) + (1-delta));
end;
initval;
k = 3.0;
c = 0.9;
end;
shocks;
var z;
periods 1;
values 0.01;
end;
perfect_foresight_setup(periods=100);
perfect_foresight_solver;
"""
    with pytest.raises(UnsupportedFeatureError) as exc_info:
        DynoModel(txt=mod_txt, filename="pf_test.mod")
    assert exc_info.value.feature in ("deterministic_shocks", "perfect_foresight")
    msg = str(exc_info.value)
    assert any(kw in msg for kw in ("periods", "perfect foresight", "deterministic"))


def test_unsupported_external_function_in_steady_state_model():
    """A user-defined function call inside steady_state_model must raise
    UnsupportedFeatureError, not a raw LARKParserError."""
    from dyno.errors import UnsupportedFeatureError

    mod_txt = """\
var y, c, k, h;
varexo e;
parameters alpha, beta, delta;
alpha = 0.36;
beta = 0.99;
delta = 0.025;
model;
c = beta * c(+1) * (alpha * k^(alpha-1) + (1-delta));
k = y - c + (1-delta)*k(-1);
y = k(-1)^alpha * h^(1-alpha) * exp(e);
h = (1-alpha)*y / c;
end;
steady_state_model;
h = my_external_helper(alpha, beta, delta);
k = h * 2.5;
y = k^alpha * h^(1-alpha);
c = y - delta*k;
end;
shocks;
var e; stderr 0.01;
end;
stoch_simul;
"""
    with pytest.raises(UnsupportedFeatureError) as exc_info:
        DynoModel(txt=mod_txt, filename="ext_ss_test.mod")
    assert exc_info.value.feature == "external_steady_state_function"
    assert "steady_state_model" in str(exc_info.value)


def test_unsupported_matlab_reporting_api():
    """MATLAB reporting API calls (dseries, report()) must raise UnsupportedFeatureError."""
    from dyno.errors import UnsupportedFeatureError

    mod_txt = """\
var y, c, k;
varexo e;
parameters alpha;
alpha = 0.33;
model;
c + k = alpha * k(-1)^alpha + (1-alpha)*k(-1) + e;
y = k(-1)^alpha + e;
c = y - k + (1-alpha)*k(-1);
end;
shocks;
var e; stderr 0.01;
end;
stoch_simul;
shock_series = dseries();
r = report();
"""
    with pytest.raises(UnsupportedFeatureError) as exc_info:
        DynoModel(txt=mod_txt, filename="reporting_test.mod")
    assert exc_info.value.feature in ("matlab_reporting_api", "macroprocessor")


# ---------------------------------------------------------------------------
# Integration tests: real mod files from the diagnostic suite
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "mod_path",
    [
        "examples/modfiles/ramst.mod",
        "examples/dynare/perfect_foresight/perfect_foresight_rbc.mod",
        "examples/dynare/perfect_foresight/perfect_foresight_expectation_errors.mod",
    ],
)
def test_perfect_foresight_mod_raises_unsupported_not_lark(mod_path):
    """Modfiles that use deterministic shocks (periods/values) must raise
    UnsupportedFeatureError, not LARKParserError."""
    from dyno.errors import UnsupportedFeatureError, LARKParserError

    try:
        DynoModel(mod_path)
    except UnsupportedFeatureError:
        pass  # expected
    except LARKParserError as e:
        pytest.fail(f"Got LARKParserError instead of UnsupportedFeatureError: {e}")
    except Exception:
        pass  # other errors (e.g. after import) are not our concern here


@pytest.mark.parametrize(
    "mod_path",
    [
        "examples/modfiles/example3.mod",
        "examples/dynare/stochastic_simulations/collard_2001_analytical_steady_state.mod",
    ],
)
def test_external_steady_state_function_mod_raises_unsupported_not_lark(mod_path):
    """Modfiles that use external function calls in steady_state_model must raise
    UnsupportedFeatureError, not LARKParserError."""
    from dyno.errors import UnsupportedFeatureError, LARKParserError

    try:
        DynoModel(mod_path)
    except UnsupportedFeatureError:
        pass  # expected
    except LARKParserError as e:
        pytest.fail(f"Got LARKParserError instead of UnsupportedFeatureError: {e}")
    except Exception:
        pass  # other errors are not our concern here


def test_example1_reporting_mod_raises_unsupported_not_lark():
    """example1_reporting.mod uses MATLAB reporting API / macroprocessor and must
    raise UnsupportedFeatureError, not LARKParserError."""
    from dyno.errors import UnsupportedFeatureError, LARKParserError

    try:
        DynoModel("examples/modfiles/example1_reporting.mod", preprocess=False)
    except UnsupportedFeatureError:
        pass  # expected
    except LARKParserError as e:
        pytest.fail(f"Got LARKParserError instead of UnsupportedFeatureError: {e}")
    except Exception:
        pass  # other errors are not our concern here


# ---------------------------------------------------------------------------
# Regression: steady_state_model blocks with intermediate scratch variables
# ---------------------------------------------------------------------------


def test_steady_state_model_intermediate_variables_are_resolved_sequentially():
    """Variables assigned earlier in a steady_state_model block must be visible
    to later lines in the same block.  Regression for the bug where intermediate
    helpers (k_H, y_H, c_H, ...) resolved to NaN because the evaluator only
    looked up self.constants, not the accumulating self.steady_states."""
    mod_txt = """\
var y, c, k, H, r, w, A;
varexo epsilon;
parameters alpha, beta, delta, gamma, phi, rho;
alpha = 0.33;
beta  = 0.984;
delta = 0.025;
gamma = 1.004;
phi   = 3.48;
rho   = 0.974;
model;
gamma = beta*c/c(+1)*(1+r(+1));
H     = 1-phi*c/w;
y     = A*k(-1)^alpha*H^(1-alpha);
r     = alpha*y/k(-1)-delta;
w     = (1-alpha)*y/H;
y     = c + gamma*k-(1-delta)*k(-1);
log(A) = rho*log(A(-1))+epsilon;
end;
steady_state_model;
A   = 1;
r   = gamma/beta-1;
k_H = (alpha/(r+delta))^(1/(1-alpha));
y_H = k_H^alpha;
w   = (1-alpha)*y_H;
c_H = y_H-(gamma-1+delta)*k_H;
H   = 1/(1+phi*c_H/w);
k   = k_H*H;
c   = c_H*H;
y   = y_H*H;
end;
shocks;
var epsilon; stderr 0.01;
end;
stoch_simul;
"""
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = DynoModel(txt=mod_txt, filename="kr2000_test.mod")

    ss = m.context["steady_states"]
    # All model variables must be finite (not NaN)
    import math

    for var in ["A", "r", "w", "H", "k", "c", "y"]:
        assert var in ss, f"Missing SS for '{var}'"
        assert not math.isnan(ss[var]), f"SS['{var}'] is NaN — sequential eval broken"

    # Sanity-check numerical values
    assert abs(ss["r"] - (1.004 / 0.984 - 1)) < 1e-10
    assert ss["H"] > 0
    assert ss["k"] > 0

    # Constants must ONLY contain parameters, not endogenous or scratch variables
    constants = m.context["constants"]
    assert set(constants.keys()) == {"alpha", "beta", "delta", "gamma", "phi", "rho"}
    assert "A" not in constants
    assert "k_H" not in constants

    # Intermediate scratch variables must NOT be in steady_states
    for scratch in ["k_H", "y_H", "c_H"]:
        assert scratch not in ss


def test_evaluate_steady_block_pure_function():
    """evaluate_steady_block must return values for endogenous variables without
    any side-effects on the input constants dictionary."""
    from dyno.dynspec.dynare import evaluate_steady_block
    from dyno.dynspec.dynare import modfile_grammar, ModFileTransformer
    from lark import Lark

    block_src = """
    steady_state_model;
    A   = 1;
    r   = gamma/beta-1;
    k_H = (alpha/(r+delta))^(1/(1-alpha));
    y_H = k_H^alpha;
    w   = (1-alpha)*y_H;
    c_H = y_H-(gamma-1+delta)*k_H;
    H   = 1/(1+phi*c_H/w);
    k   = k_H*H;
    c   = c_H*H;
    y   = y_H*H;
    end;
    """
    full_mod = f"""
    var y, c, k, H, r, w, A;
    varexo epsilon;
    parameters alpha, beta, delta, gamma, phi, rho;
    model;
    c = c(+1);
    end;
    {block_src}
    """
    trans = ModFileTransformer()
    parser = Lark(modfile_grammar, parser="lalr", transformer=trans)
    tree = parser.parse(full_mod)

    # Find the steady_block node
    steady_tree = next(
        ch for ch in tree.children if getattr(ch, "data", None) == "steady_block"
    )

    constants_input = {
        "alpha": 0.33,
        "beta": 0.984,
        "delta": 0.025,
        "gamma": 1.004,
        "phi": 3.48,
        "rho": 0.974,
    }
    constants_copy = dict(constants_input)
    endogenous = ["y", "c", "k", "H", "r", "w", "A"]

    result = evaluate_steady_block(
        steady_tree,
        constants=constants_input,
        endogenous=endogenous,
    )

    # 1. Returned dictionary must only contain endogenous variables
    assert set(result.keys()) == set(endogenous)
    for scratch in ["k_H", "y_H", "c_H"]:
        assert scratch not in result

    # 2. Input constants dictionary must NOT have been modified
    assert constants_input == constants_copy

    # 3. Values must be finite and match expectations
    assert result["A"] == 1.0
    assert abs(result["r"] - (1.004 / 0.984 - 1)) < 1e-10
    assert result["k"] > 0
    assert result["c"] > 0


@pytest.mark.parametrize(
    "mod_path",
    [
        "examples/modfiles/model_KR2000_IRF.mod",
        "examples/modfiles/model_KR2000_STAT.mod",
    ],
)
def test_kr2000_models_run_without_typeerror(mod_path):
    """model_KR2000_*.mod must import and run without a TypeError.
    Previously, intermediate variables in steady_state_model resolved to NaN,
    causing the numerical solver to produce complex residuals."""
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = DynoModel(mod_path)
        results = m.run(default_pipeline=True)

    import numpy as np

    assert results is not None
    assert (
        np.max(np.abs(results.residuals)) < 1e-8
    ), f"Steady-state residuals too large: {results.residuals}"
