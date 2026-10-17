"""Tests for recipes files, `check_group` and `check_model`."""

import pytest

from dyno.dynspec import (
    DTCC_RECIPE,
    check_group,
    check_model,
    extract_complementarity,
    get_recipe,
    load_recipes,
    parser,
)

VARIABLES = {
    "exogenous": ["e_z"],
    "states": ["z", "k"],
    "controls": ["n", "i"],
    "auxiliaries": ["y", "c"],
}

with open("examples/rbc_dolo_comp.dyno", encoding="utf-8") as f:
    RBC_COMP = f.read()


def messages(result):
    return [v.message for v in result.violations]


# ── Recipes files ────────────────────────────────────────────────────────────


def test_bundled_recipe():
    assert get_recipe("dtcc") is DTCC_RECIPE
    assert get_recipe(DTCC_RECIPE) is DTCC_RECIPE
    assert DTCC_RECIPE.process_group == "exogenous"
    arb = DTCC_RECIPE.get_group_spec("arbitrage")
    assert arb.n_equations == "controls"
    assert (
        arb.complementarities.variable.group,
        arb.complementarities.variable.shift,
    ) == ("controls", 0)
    bounds = DTCC_RECIPE.get_group_spec("arbitrage_bounds")
    assert {(v.group, v.shift) for v in bounds.allowed} == {
        ("exogenous", 0),
        ("states", 0),
        ("parameters", 0),
    }
    assert bounds.optional


def test_unknown_recipe():
    with pytest.raises(ValueError, match="Unknown recipe 'nope'"):
        get_recipe("nope")


def test_load_recipes_from_file(tmp_path):
    f = tmp_path / "recipes.yaml"
    f.write_text("""
simple:
    symbols: [states, shocks, parameters]
    specs:
        law:
            target: [states, 0, s]
            eqs:
                - [states, -1]
                - [shocks, 0]
""")
    recipe = load_recipes(f)["simple"]
    spec = recipe.get_group_spec("law")
    assert spec.target == "states" and spec.recursive and spec.n_equations == "states"
    assert not spec.optional and spec.complementarities is None

    ok = check_group(
        "x[t] = x[t-1] + e[t]", "law", {"states": ["x"], "shocks": ["e"]}, recipe=recipe
    )
    assert ok
    bad = check_group("x[t] = x[t+1]", "law", {"states": ["x"]}, recipe=recipe)
    assert not bad


# ── check_group ──────────────────────────────────────────────────────────────


def test_check_group_from_text():
    result = check_group(
        """
        z[t] = rho*z[t-1] + e_z[t]
        k[t] = (1-delta)*k[t-1] + i[t-1]
        """,
        "transition",
        VARIABLES,
    )
    assert result.ok, str(result)
    assert result.dag_order == ["z", "k"]


def test_check_group_reports_lines():
    result = check_group(
        ["z[t] = rho*z[t-1] + e_z[t]", "k[t] = (1-delta)*k[t-1] + i[t]"],
        "transition",
        VARIABLES,
    )
    assert not result.ok
    (v,) = result.violations
    assert (v.equation_index, v.line, v.variable_name, v.shift) == (1, 2, "i", 0)
    assert "Equation 1 (line 2): i[t]" in str(result)


def test_check_group_accepts_trees():
    trees = parser.parse(
        "z[t] = rho*z[t-1] + e_z[t]\n", start="equation_block"
    ).children
    assert not check_group(trees, "transition", VARIABLES)  # one equation, two states


def test_check_group_unknown_group():
    with pytest.raises(ValueError, match="no equation group 'arbitrages'"):
        check_group("a[t]", "arbitrages", VARIABLES)


def test_complementarities_conform():
    result = check_group(
        """
        c[t] - w ⟂ 0 <= n[t] <= k[t]
        1 - c[t+1] ⟂ -inf <= i[t] <= e_z[t] + kmax
        """,
        "arbitrage",
        VARIABLES,
    )
    assert result.ok, str(result)


def test_complementarity_bounds_the_wrong_control():
    result = check_group(
        """
        c[t] ⟂ 0 <= i[t]
        c[t+1] ⟂ 0 <= n[t]
        """,
        "arbitrage",
        VARIABLES,
    )
    assert messages(result) == [
        "expected the complementarity to bound 'n' ('controls' variable #1), found 'i'",
        "expected the complementarity to bound 'i' ('controls' variable #2), found 'n'",
    ]


@pytest.mark.parametrize("bound", ["c[t]", "n[t]", "k[t-1]", "z[t+1]"])
def test_complementarity_bounds_restricted(bound):
    result = check_group(
        f"""
        c[t] ⟂ 0 <= n[t] <= {bound}
        c[t+1] ⟂ 0 <= i[t]
        """,
        "arbitrage",
        VARIABLES,
    )
    (v,) = result.violations
    assert v.equation_index == 0
    assert "bounds of 'arbitrage' complementarities" in v.message


def test_complementarity_must_bound_a_current_control():
    result = check_group(
        "c[t] ⟂ 0 <= k[t]", "arbitrage", {"states": ["k"], "controls": []}
    )
    assert any("must bound a 'controls' variable" in m for m in messages(result))


def test_complementarity_not_allowed_in_transition():
    result = check_group(
        """
        z[t] = rho*z[t-1] + e_z[t]
        k[t] - i[t-1] ⟂ 0 <= k[t]
        """,
        "transition",
        VARIABLES,
    )
    assert any(
        "complementarity conditions are not allowed" in m for m in messages(result)
    )


def test_standalone_bounds_group():
    assert check_group("n[t] = 0\ni[t] = k[t] + e_z[t]", "controls_ub", VARIABLES)
    result = check_group("n[t] = 0\ni[t] = c[t]", "controls_ub", VARIABLES)
    assert messages(result) == [
        "'auxiliaries' variables at shift +0 are not allowed in 'controls_ub' equations"
    ]


def test_extract_complementarity():
    tree = parser.parse("a[t] ⟂ k[t-1] <= x[t]\n", start="equation_block").children[0]
    comp = extract_complementarity(tree)
    assert (comp.variable, comp.shift) == ("x", 0)
    assert comp.lower.data == "variable" and comp.upper.data == "constant"
    assert extract_complementarity(tree.children[0]) is None


# ── check_model ──────────────────────────────────────────────────────────────


def test_check_model_rbc_dolo_comp():
    result = check_model(RBC_COMP)
    assert result.ok, str(result)
    assert result.variables == {
        "exogenous": ["e_z"],
        "states": ["z", "k"],
        "controls": ["n", "i"],
        "auxiliaries": ["y", "c", "rk", "w"],
    }
    assert set(result.groups) == {"definitions", "transition", "arbitrage"}
    assert result.groups["transition"].dag_order == ["z", "k"]


def test_check_model_accepts_a_tree():
    assert check_model(parser.parse(RBC_COMP, start="free_block"))


def test_check_model_given_controls_order():
    result = check_model(RBC_COMP, variables={"controls": ["i", "n"]})
    assert not result.ok
    assert "expected the complementarity to bound 'i'" in str(result)


def test_check_model_standalone_bounds_sections():
    model = RBC_COMP.replace(
        "@run: steady",
        """
[controls_lb] :: {
    n[t] = 0.0
    i[t] = -inf
}
[controls_ub] :: {
    n[t] = inf
    i[t] = (1-delta)*k[t] + exp(z[t])
}
@run: steady""",
    )
    result = check_model(model)
    assert result.ok, str(result)
    assert set(result.groups) >= {"controls_lb", "controls_ub"}

    bad = check_model(model.replace("i[t] = -inf", "i[t] = -c[t]"))
    assert messages(bad.groups["controls_lb"]) == [
        "'auxiliaries' variables at shift +0 are not allowed in 'controls_lb' equations"
    ]


def test_check_model_structure_violations():
    model = """
rho <- 0.9
e[t] <- N(0, 0.01)
[transition] :: {
    z[t] = rho*z[t-1] + e[t]
}
x[t] = z[t]
y[t] = z[t] :: [definitions, arbitrage]
"""
    result = check_model(model)
    assert not result.ok
    found = [str(v) for v in result.violations]
    assert found == [
        "Line 7: equation is not tagged with an equation group of 'dtcc' "
        "(definitions, transition, arbitrage, arbitrage_bounds, controls_lb, controls_ub)",
        "Line 8: equation is tagged with several equation groups (definitions, arbitrage)",
        "missing required equation group 'arbitrage'",
    ]
    assert result.groups["transition"].ok
    assert result.variables == {"exogenous": ["e"], "states": ["z"]}
