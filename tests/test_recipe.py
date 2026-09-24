"""Tests for the recipe conformity checker and function generator."""

import math

import numpy as np
import pytest

from dyno import DynoModel
from dyno.larkfiles import DynoFile
from dyno.dynspec.grammar import parser
from dyno.dynspec.recipe import (
    Recipe,
    EquationGroupSpec,
    VariableSpec,
    DTCC_RECIPE,
    ConformityResult,
    extract_variables_from_equation,
    extract_lhs_variable,
    extract_rhs_variables,
    check_equation_group,
    check_dag,
)
from dyno.dynspec.funcgen import (
    DefinitionsBlock,
    build_definitions_block,
    generate_equation_function,
    compile_equation_group,
)

# ── Helpers ──────────────────────────────────────────────────────────────────


def _parse_eq(text: str):
    """Parse a single equation string into a Lark Tree."""
    return parser.parse(text, start="formula")


def _parse_equality(text: str):
    """Parse 'lhs = rhs' into an equality Tree."""
    # The grammar expects equation_block or free_block for equalities.
    # We parse a free_block with one equation.
    block = parser.parse(text, start="free_block")
    # Navigate to the actual equality/bare_formula node
    for child in block.iter_subtrees():
        if child.data in ("equality", "bare_formula"):
            return child
    raise ValueError(f"Could not parse equality from: {text}")


# ── extract_variables_from_equation ──────────────────────────────────────────


class TestExtractVariables:
    def test_simple_variable(self):
        eq = _parse_equality("x[t] = a")
        result = extract_variables_from_equation(eq)
        assert "x" in result
        assert 0 in result["x"]

    def test_variable_with_shift(self):
        eq = _parse_equality("k[t] = k[t-1] + i[t-1]")
        result = extract_variables_from_equation(eq)
        assert "k" in result
        assert result["k"] == {0, -1}
        assert "i" in result
        assert result["i"] == {-1}

    def test_forward_looking(self):
        eq = _parse_equality("x[t] = y[t+1] + z[t]")
        result = extract_variables_from_equation(eq)
        assert result["y"] == {1}
        assert result["z"] == {0}
        assert result["x"] == {0}

    def test_bare_formula(self):
        eq = _parse_equality("c[t]^2 - y[t]")
        result = extract_variables_from_equation(eq)
        assert "c" in result
        assert result["c"] == {0}
        assert "y" in result

    def test_nested_function_calls(self):
        eq = _parse_equality("z[t] = exp(z[t-1]) + log(k[t])")
        result = extract_variables_from_equation(eq)
        assert result["z"] == {0, -1}
        assert result["k"] == {0}

    def test_constants_not_included(self):
        eq = _parse_equality("y[t] = alpha*k[t-1]")
        result = extract_variables_from_equation(eq)
        assert "y" in result
        assert "k" in result
        # alpha is a constant (no time index), should NOT appear
        assert "alpha" not in result


class TestExtractLhsVariable:
    def test_equality_lhs(self):
        eq = _parse_equality("k[t] = (1-delta)*k[t-1]")
        assert extract_lhs_variable(eq) == "k"

    def test_bare_formula_returns_none(self):
        eq = _parse_equality("c[t]^2 - y[t]")
        assert extract_lhs_variable(eq) is None


# ── DTCC Recipe Structure ────────────────────────────────────────────────────


class TestDTCCRecipe:
    def test_has_expected_groups(self):
        assert DTCC_RECIPE.name == "dtcc"
        assert "exogenous" in DTCC_RECIPE.variable_groups
        assert "states" in DTCC_RECIPE.variable_groups
        assert "controls" in DTCC_RECIPE.variable_groups
        assert "auxiliaries" in DTCC_RECIPE.variable_groups
        assert "parameters" in DTCC_RECIPE.variable_groups

    def test_definitions_spec(self):
        spec = DTCC_RECIPE.get_group_spec("definitions")
        assert spec is not None
        assert spec.target == "auxiliaries"
        assert spec.recursive is True
        assert spec.optional is True
        allowed_pairs = {(vs.group, vs.shift) for vs in spec.allowed}
        assert ("exogenous", 0) in allowed_pairs
        assert ("states", 0) in allowed_pairs
        assert ("controls", 0) in allowed_pairs
        assert ("auxiliaries", 0) in allowed_pairs
        assert ("parameters", 0) in allowed_pairs

    def test_transition_spec(self):
        spec = DTCC_RECIPE.get_group_spec("transition")
        assert spec is not None
        assert spec.target == "states"
        assert spec.recursive is True
        allowed_pairs = {(vs.group, vs.shift) for vs in spec.allowed}
        assert ("states", -1) in allowed_pairs
        assert ("controls", -1) in allowed_pairs
        assert ("exogenous", 0) in allowed_pairs
        # Should NOT allow controls at t
        assert ("controls", 0) not in allowed_pairs

    def test_arbitrage_spec(self):
        spec = DTCC_RECIPE.get_group_spec("arbitrage")
        assert spec is not None
        assert spec.target is None
        assert spec.recursive is False
        allowed_pairs = {(vs.group, vs.shift) for vs in spec.allowed}
        assert ("controls", 0) in allowed_pairs
        assert ("controls", 1) in allowed_pairs
        assert ("states", 0) in allowed_pairs
        assert ("states", 1) in allowed_pairs
        # Auxiliaries allowed at t and t+1
        assert ("auxiliaries", 0) in allowed_pairs
        assert ("auxiliaries", 1) in allowed_pairs


# ── Conformity Checking ─────────────────────────────────────────────────────


class TestCheckEquationGroup:
    """Tests for check_equation_group."""

    def _variables(self):
        return {
            "exogenous": ["e_z"],
            "states": ["z", "k"],
            "controls": ["n", "i"],
        }

    def test_valid_transition(self):
        """Valid transition equations should pass."""
        txt = """\
z[t] = rho*z[t-1] + e_z[t]
k[t] = (1-delta)*k[t-1] + i[t-1]
"""
        block = parser.parse(txt, start="equation_block")
        equations = [
            child
            for child in block.iter_subtrees()
            if child.data in ("equality", "bare_formula")
        ]

        spec = DTCC_RECIPE.get_group_spec("transition")
        assert spec is not None
        result = check_equation_group(
            equations,
            self._variables(),
            spec,
            constants=["rho", "delta"],
        )
        assert result.ok
        assert result.dag_order is not None
        assert set(result.dag_order) == {"z", "k"}

    def test_invalid_transition_future_variable(self):
        """Transition with a t+1 variable should fail."""
        txt = """\
z[t] = rho*z[t-1] + k[t+1]
k[t] = k[t-1]
"""
        block = parser.parse(txt, start="equation_block")
        equations = [
            child
            for child in block.iter_subtrees()
            if child.data in ("equality", "bare_formula")
        ]

        spec = DTCC_RECIPE.get_group_spec("transition")
        assert spec is not None
        result = check_equation_group(
            equations,
            self._variables(),
            spec,
            constants=["rho"],
        )
        assert not result.ok
        assert any("shift +1" in str(v) for v in result.violations)

    def test_invalid_transition_controls_at_t(self):
        """Transition with controls at t (not t-1) should fail."""
        txt = """\
z[t] = z[t-1]
k[t] = k[t-1] + i[t]
"""
        block = parser.parse(txt, start="equation_block")
        equations = [
            child
            for child in block.iter_subtrees()
            if child.data in ("equality", "bare_formula")
        ]

        spec = DTCC_RECIPE.get_group_spec("transition")
        assert spec is not None
        result = check_equation_group(
            equations,
            self._variables(),
            spec,
            constants=[],
        )
        assert not result.ok
        violations_str = "\n".join(str(v) for v in result.violations)
        assert "controls" in violations_str

    def test_valid_arbitrage(self):
        """Valid arbitrage equations (bare formulas) should pass."""
        txt = """\
n[t] - z[t]
i[t] - k[t] + k[t+1]
"""
        block = parser.parse(txt, start="equation_block")
        equations = [
            child
            for child in block.iter_subtrees()
            if child.data in ("equality", "bare_formula")
        ]

        spec = DTCC_RECIPE.get_group_spec("arbitrage")
        assert spec is not None
        result = check_equation_group(
            equations,
            self._variables(),
            spec,
            constants=[],
        )
        assert result.ok

    def test_invalid_arbitrage_lag(self):
        """Arbitrage with t-1 variables should fail."""
        txt = """\
n[t] - z[t-1]
i[t] - k[t]
"""
        block = parser.parse(txt, start="equation_block")
        equations = [
            child
            for child in block.iter_subtrees()
            if child.data in ("equality", "bare_formula")
        ]

        spec = DTCC_RECIPE.get_group_spec("arbitrage")
        assert spec is not None
        result = check_equation_group(
            equations,
            self._variables(),
            spec,
            constants=[],
        )
        assert not result.ok

    def test_wrong_equation_count(self):
        """Wrong number of transition equations should report a violation."""
        txt = "z[t] = z[t-1]"
        block = parser.parse(txt, start="equation_block")
        equations = [
            child
            for child in block.iter_subtrees()
            if child.data in ("equality", "bare_formula")
        ]
        # states has 2 vars but we only have 1 equation
        spec = DTCC_RECIPE.get_group_spec("transition")
        assert spec is not None
        result = check_equation_group(
            equations,
            self._variables(),
            spec,
            constants=[],
        )
        assert not result.ok
        assert any("Expected 2 equations" in str(v) for v in result.violations)


# ── DAG Checking ─────────────────────────────────────────────────────────────


class TestCheckDAG:
    def test_valid_dag_independent(self):
        """Two independent transition equations form a trivial DAG."""
        txt = """\
z[t] = z[t-1]
k[t] = k[t-1]
"""
        block = parser.parse(txt, start="equation_block")
        equations = [
            child
            for child in block.iter_subtrees()
            if child.data in ("equality", "bare_formula")
        ]
        variables = {"states": ["z", "k"], "controls": ["n", "i"]}

        order = check_dag(equations, "states", variables)
        assert order is not None
        assert set(order) == {"z", "k"}

    def test_valid_dag_dependent(self):
        """Sequential dependency: k depends on z at t -> z must come first."""
        txt = """\
z[t] = z[t-1]
k[t] = z[t] + k[t-1]
"""
        block = parser.parse(txt, start="equation_block")
        equations = [
            child
            for child in block.iter_subtrees()
            if child.data in ("equality", "bare_formula")
        ]
        variables = {"states": ["z", "k"]}

        order = check_dag(equations, "states", variables)
        assert order is not None
        assert order.index("z") < order.index("k")

    def test_cycle_detected(self):
        """Mutual dependency: z depends on k[t] and k depends on z[t] -> cycle."""
        txt = """\
z[t] = k[t]
k[t] = z[t]
"""
        block = parser.parse(txt, start="equation_block")
        equations = [
            child
            for child in block.iter_subtrees()
            if child.data in ("equality", "bare_formula")
        ]
        variables = {"states": ["z", "k"]}

        order = check_dag(equations, "states", variables)
        assert order is None


# ── RBC Model Integration ───────────────────────────────────────────────────


class TestRBCDoloModel:
    """Integration tests using the RBC dolo model file."""

    @pytest.fixture
    def model(self):
        return DynoModel("examples/rbc_dolo.dyno")

    def test_model_loads(self, model):
        assert len(model.equations) == 8

    def test_residuals_near_zero(self, model):
        residuals = model.residuals
        assert all(abs(r) < 1e-10 for r in residuals)

    def test_transition_equations_tagged(self, model):
        tagged = model.equations_with_tags()
        transition_eqs = [
            (i, text, tags) for i, text, tags in tagged if "transition" in tags
        ]
        assert len(transition_eqs) == 2

    def test_arbitrage_equations_tagged(self, model):
        tagged = model.equations_with_tags()
        arb_eqs = [(i, text, tags) for i, text, tags in tagged if "arbitrage" in tags]
        assert len(arb_eqs) == 2

    def test_definitions_equations_tagged(self, model):
        tagged = model.equations_with_tags()
        def_eqs = [(i, text, tags) for i, text, tags in tagged if "definitions" in tags]
        assert len(def_eqs) == 4

    def test_transition_conforms_to_dtcc(self, model):
        """Transition equations should pass DTCC recipe check."""
        trans_eqs = []
        for eq in model.symbolic.equations:
            meta = getattr(getattr(eq, "meta", None), "statement_metadata", {})
            if "transition" in meta.get("tags", []):
                trans_eqs.append(eq)

        variables = {
            "exogenous": ["e_z"],
            "states": ["z", "k"],
            "controls": ["n", "i"],
            "auxiliaries": ["y", "c", "rk", "w"],
        }

        spec = DTCC_RECIPE.get_group_spec("transition")
        assert spec is not None
        result = check_equation_group(
            trans_eqs,
            variables,
            spec,
            constants=list(model.context["constants"].keys()),
        )
        assert result.ok, f"Transition conformity failed:\n{result}"
        assert result.dag_order is not None
        assert set(result.dag_order) == {"z", "k"}

    def test_definitions_conform_to_dtcc(self, model):
        """Definition equations should pass DTCC recipe check."""
        def_eqs = []
        for eq in model.symbolic.equations:
            meta = getattr(getattr(eq, "meta", None), "statement_metadata", {})
            if "definitions" in meta.get("tags", []):
                def_eqs.append(eq)

        variables = {
            "exogenous": ["e_z"],
            "states": ["z", "k"],
            "controls": ["n", "i"],
            "auxiliaries": ["y", "c", "rk", "w"],
        }

        spec = DTCC_RECIPE.get_group_spec("definitions")
        assert spec is not None
        result = check_equation_group(
            def_eqs,
            variables,
            spec,
            constants=list(model.context["constants"].keys()),
        )
        assert result.ok, f"Definitions conformity failed:\n{result}"
        assert result.dag_order is not None
        # y must come before c, rk, w (they all depend on y)
        assert result.dag_order.index("y") < result.dag_order.index("c")
        assert result.dag_order.index("y") < result.dag_order.index("rk")
        assert result.dag_order.index("y") < result.dag_order.index("w")

    def test_arbitrage_conforms_to_dtcc(self, model):
        """Arbitrage equations should pass DTCC recipe check."""
        arb_eqs = []
        for eq in model.symbolic.equations:
            meta = getattr(getattr(eq, "meta", None), "statement_metadata", {})
            if "arbitrage" in meta.get("tags", []):
                arb_eqs.append(eq)

        variables = {
            "exogenous": ["e_z"],
            "states": ["z", "k"],
            "controls": ["n", "i"],
            "auxiliaries": ["y", "c", "rk", "w"],
        }

        spec = DTCC_RECIPE.get_group_spec("arbitrage")
        assert spec is not None
        result = check_equation_group(
            arb_eqs,
            variables,
            spec,
            constants=list(model.context["constants"].keys()),
        )
        assert result.ok, f"Arbitrage conformity failed:\n{result}"


# ── Function Generation ─────────────────────────────────────────────────────


class TestFunctionGeneration:
    """Tests for the function generator."""

    @pytest.fixture
    def model(self):
        return DynoModel("examples/rbc_dolo.dyno")

    def _get_equations_by_tag(self, model, tag):
        eqs = []
        for eq in model.symbolic.equations:
            meta = getattr(getattr(eq, "meta", None), "statement_metadata", {})
            if tag in meta.get("tags", []):
                eqs.append(eq)
        return eqs

    def _rbc_variables(self):
        return {
            "exogenous": ["e_z"],
            "states": ["z", "k"],
            "controls": ["n", "i"],
            "auxiliaries": ["y", "c", "rk", "w"],
        }

    def test_compile_transition(self, model):
        """Compile and evaluate transition function at steady state."""
        trans_eqs = self._get_equations_by_tag(model, "transition")
        variables = self._rbc_variables()
        spec = DTCC_RECIPE.get_group_spec("transition")
        assert spec is not None

        result, func = compile_equation_group(
            trans_eqs,
            variables,
            spec,
            constants=model.context["constants"],
        )
        assert result.ok
        assert func is not None

        # Evaluate at steady state
        ss = model.steady_state
        exo_m1 = np.array([ss.get("e_z", 0.0)])
        states_m1 = np.array([ss["z"], ss["k"]])
        controls_m1 = np.array([ss["n"], ss["i"]])
        exo_0 = np.array([ss.get("e_z", 0.0)])

        # transition spec: exo[-1], states[-1], controls[-1], exo[0]
        new_states = func(exo_m1, states_m1, controls_m1, exo_0)

        assert new_states.shape == (2,)
        # At steady state, transition should return steady state values
        np.testing.assert_allclose(new_states[0], ss["z"], atol=1e-10)
        np.testing.assert_allclose(new_states[1], ss["k"], atol=1e-10)

    def test_compile_arbitrage_with_definitions(self, model):
        """Compile arbitrage function with definitions auto-inlined."""
        arb_eqs = self._get_equations_by_tag(model, "arbitrage")
        def_eqs = self._get_equations_by_tag(model, "definitions")
        variables = self._rbc_variables()

        # Build definitions block
        def_spec = DTCC_RECIPE.get_group_spec("definitions")
        assert def_spec is not None
        def_result, def_block = build_definitions_block(
            def_eqs,
            variables,
            def_spec,
            constants=list(model.context["constants"].keys()),
        )
        assert def_result.ok
        assert def_block is not None

        # Compile arbitrage with definitions
        arb_spec = DTCC_RECIPE.get_group_spec("arbitrage")
        assert arb_spec is not None
        result, func = compile_equation_group(
            arb_eqs,
            variables,
            arb_spec,
            constants=model.context["constants"],
            definitions=def_block,
        )
        assert result.ok
        assert func is not None

        # Evaluate at steady state
        ss = model.steady_state
        exo_0 = np.array([ss.get("e_z", 0.0)])
        states_0 = np.array([ss["z"], ss["k"]])
        controls_0 = np.array([ss["n"], ss["i"]])
        exo_1 = np.array([ss.get("e_z", 0.0)])
        states_1 = np.array([ss["z"], ss["k"]])
        controls_1 = np.array([ss["n"], ss["i"]])

        # auxiliaries are NOT passed as arguments — they are auto-computed
        # arg order: exo[0], states[0], controls[0], exo[1], states[1], controls[1]
        residuals = func(exo_0, states_0, controls_0, exo_1, states_1, controls_1)

        assert residuals.shape == (2,)
        np.testing.assert_allclose(residuals, 0.0, atol=1e-10)

    def test_definitions_block_topological_order(self, model):
        """build_definitions_block should return correct topological order."""
        def_eqs = self._get_equations_by_tag(model, "definitions")
        variables = self._rbc_variables()
        def_spec = DTCC_RECIPE.get_group_spec("definitions")
        assert def_spec is not None

        result, block = build_definitions_block(
            def_eqs,
            variables,
            def_spec,
            constants=list(model.context["constants"].keys()),
        )
        assert result.ok
        assert block is not None
        # y must come before c (c = y - i)
        assert block.variable_names.index("y") < block.variable_names.index("c")
        assert block.target_group == "auxiliaries"
        assert len(block.equations) == 4

    def test_compile_rejects_nonconforming(self):
        """compile_equation_group should return None for non-conforming equations."""
        txt = "z[t] = z[t+1]"
        block = parser.parse(txt, start="equation_block")
        equations = [
            child
            for child in block.iter_subtrees()
            if child.data in ("equality", "bare_formula")
        ]
        variables = {"states": ["z", "k"], "controls": ["n"]}
        spec = DTCC_RECIPE.get_group_spec("transition")
        assert spec is not None

        result, func = compile_equation_group(equations, variables, spec)
        assert not result.ok
        assert func is None


# ── ConformityResult Display ─────────────────────────────────────────────────


class TestConformityResultDisplay:
    def test_ok_str(self):
        r = ConformityResult(group_name="transition", ok=True, dag_order=["z", "k"])
        s = str(r)
        assert "✓" in s
        assert "z → k" in s

    def test_failure_str(self):
        from dyno.dynspec.recipe import Violation

        r = ConformityResult(
            group_name="arbitrage",
            ok=False,
            violations=[
                Violation(
                    equation_index=0,
                    variable_name="k",
                    shift=-1,
                    message="states at shift -1 not allowed",
                )
            ],
        )
        s = str(r)
        assert "✗" in s
        assert "k[t-1]" in s

    def test_bool_conversion(self):
        assert bool(ConformityResult(group_name="test", ok=True))
        assert not bool(ConformityResult(group_name="test", ok=False))
