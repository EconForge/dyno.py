import os
import shutil
import subprocess
import pytest
from dyno.dynare import (
    expand_macro,
    macroexpand,
    MacroProcessor,
    MacroEnvironment,
    MacroError,
    MacroSyntaxError,
    MacroEvaluationError,
)

PREPROCESSOR_BIN = shutil.which("dynare-preprocessor")
if not PREPROCESSOR_BIN:
    for env in ["dynare", "dev-dynare"]:
        cand = os.path.expanduser(
            f"/home/pablo/Econforge/dyno.py/.pixi/envs/{env}/bin/dynare-preprocessor"
        )
        if os.path.exists(cand):
            PREPROCESSOR_BIN = cand
            break


def run_dynare_savemacro(mod_path: str, tmp_path) -> str:
    """Runs official dynare-preprocessor savemacro and returns output string."""
    assert PREPROCESSOR_BIN is not None, "dynare-preprocessor binary not found"
    out_file = str(tmp_path / "dynare_out.mod")
    res = subprocess.run(
        [PREPROCESSOR_BIN, mod_path, "onlymacro", f"savemacro={out_file}"],
        capture_output=True,
        text=True,
    )
    if res.returncode != 0:
        raise RuntimeError(f"dynare-preprocessor failed: {res.stderr}")
    with open(out_file, "r", encoding="utf-8") as f:
        return f.read()


class TestMacroExpressions:
    def test_arithmetic(self):
        txt = "@#define x = 2 + 3 * 4 ^ 2\n@{x}"
        assert expand_macro(txt).strip() == "50"

    def test_float_formatting(self):
        txt = "@#define f1 = 0.025\n@#define f2 = 1e-5\n@{f1} @{f2}"
        out = expand_macro(txt).strip()
        assert out == "0.025000000000000001 1.0000000000000001e-05"

    def test_booleans_and_logic(self):
        txt = (
            "@#define b1 = true && false\n"
            "@#define b2 = true || false\n"
            "@#define b3 = !b1\n"
            "@{b1} @{b2} @{b3}"
        )
        assert expand_macro(txt).strip() == "false true true"

    def test_short_circuit_logic(self):
        txt = "@#define t = true || undefined_var\n@{t}"
        assert expand_macro(txt).strip() == "true"
        txt2 = "@#define f = false && undefined_var\n@{f}"
        assert expand_macro(txt2).strip() == "false"

    def test_real_condition_coercion(self):
        txt = "@#if 1\nYES\n@#else\nNO\n@#endif"
        assert expand_macro(txt).strip() == "YES"
        txt0 = "@#if 0\nYES\n@#else\nNO\n@#endif"
        assert expand_macro(txt0).strip() == "NO"

    def test_relational_and_membership(self):
        txt = (
            "@#define a = [1, 2, 3]\n"
            "@#define in_a = 2 in a\n"
            "@#define not_in_a = 5 in a\n"
            "@{in_a} @{not_in_a}"
        )
        assert expand_macro(txt).strip() == "true false"

    def test_boolean_comparisons(self):
        txt = "@#define x = 10 > 5\n@{x}"
        assert expand_macro(txt).strip() == "true"

    def test_range_generation(self):
        txt = "@#define r1 = 1:4\n@#define r2 = 4:-1:1\n@#define r3 = 4:1\n@{r1} @{r2} @{r3}"
        assert expand_macro(txt).strip() == "[1, 2, 3, 4] [4, 3, 2, 1] []"

    def test_cartesian_product_and_power(self):
        txt = (
            "@#define a = [1, 2]\n"
            "@#define b = [3, 4]\n"
            "@#define prod = a * b\n"
            "@#define pow2 = a ^ 2\n"
            "@{prod}\n@{pow2}"
        )
        lines = expand_macro(txt).strip().splitlines()
        assert lines[0] == "[(1, 3), (1, 4), (2, 3), (2, 4)]"
        assert lines[1] == "[(1, 1), (1, 2), (2, 1), (2, 2)]"

    def test_set_union_and_intersection(self):
        txt = (
            "@#define a = [1, 2, 3]\n"
            "@#define b = [2, 3, 4]\n"
            "@#define u = a | b\n"
            "@#define i = a & b\n"
            "@#define d = a - b\n"
            "@{u}\n@{i}\n@{d}"
        )
        lines = expand_macro(txt).strip().splitlines()
        assert lines[0] == "[1, 2, 3, 4]"
        assert lines[1] == "[2, 3]"
        assert lines[2] == "[1]"

    def test_comprehensions(self):
        txt = (
            "@#define a = [1, 2, 3, 4]\n"
            "@#define evens = [x in a when x > 2]\n"
            "@#define squares = [x^2 for x in a when x <= 3]\n"
            "@{evens}\n@{squares}"
        )
        lines = expand_macro(txt).strip().splitlines()
        assert lines[0] == "[3, 4]"
        assert lines[1] == "[1, 4, 9]"

    def test_comprehension_with_tuples(self):
        txt = (
            "@#define pairs = [(1, 10), (2, 20), (3, 30)]\n"
            "@#define sums = [x + y for (x, y) in pairs when x > 1]\n"
            "@{sums}"
        )
        assert expand_macro(txt).strip() == "[22, 33]"

    def test_builtins(self):
        txt = (
            "@#define s = sum([1, 2, 3, 4])\n"
            "@#define p = prod([2, 3, 4])\n"
            "@#define d = diff([10, 2, 3])\n"
            "@#define l = length([10, 20, 30])\n"
            "@#define emp = isEmpty([])\n"
            "@#define m = max([1, 5, 3])\n"
            "@#define mn = min([1, 5, 3])\n"
            "@#define rnd = round(2.5)\n"
            "@#define rnd_neg = round(-2.5)\n"
            "@{s} @{p} @{d} @{l} @{emp} @{m} @{mn} @{rnd} @{rnd_neg}"
        )
        assert expand_macro(txt).strip() == "10 24 5 3 true 5 1 3 -3"

    def test_defined_builtin(self):
        txt = (
            "@#define a = 1\n"
            "@#define has_a = defined(a)\n"
            "@#define has_b = defined(b)\n"
            "@{has_a} @{has_b}"
        )
        assert expand_macro(txt).strip() == "true false"


class TestMacroDirectives:
    def test_line_continuation(self):
        txt = "@#define long_list = [1, \\\\\n" "2, \\\\\n" "3]\n" "var @{long_list};"
        assert expand_macro(txt).strip() == "var [1, 2, 3];"

    def test_for_loop(self):
        txt = (
            '@#define countries = ["H", "F"]\n'
            "@#for c in countries\n"
            "var Y_@{c};\n"
            "@#endfor"
        )
        out = expand_macro(txt).strip().splitlines()
        assert out == ["var Y_H;", "var Y_F;"]

    def test_for_loop_with_tuples(self):
        txt = (
            '@#define pairs = [("A", 1), ("B", 2)]\n'
            "@#for (c, num) in pairs\n"
            "var @{c}_@{num};\n"
            "@#endfor"
        )
        out = expand_macro(txt).strip().splitlines()
        assert out == ["var A_1;", "var B_2;"]

    def test_macro_functions(self):
        txt = (
            "@#define add(x, y) = x + y\n"
            "@#define mult(x, y) = x * y\n"
            "res1 = @{add(10, 20)};\n"
            "res2 = @{mult(5, 6)};"
        )
        out = expand_macro(txt).strip().splitlines()
        assert out == ["res1 = 30;", "res2 = 30;"]

    def test_if_elif_else(self):
        txt = (
            "@#define mode = 2\n"
            "@#if mode == 1\n"
            "ONE\n"
            "@#elif mode == 2\n"
            "TWO\n"
            "@#else\n"
            "THREE\n"
            "@#endif"
        )
        assert expand_macro(txt).strip() == "TWO"

    def test_ifdef_ifndef(self):
        txt = (
            "@#ifndef VAR\n"
            "@#define VAR = 10\n"
            "@#endif\n"
            "@#ifdef VAR\n"
            "DEFINED: @{VAR}\n"
            "@#endif"
        )
        assert expand_macro(txt).strip() == "DEFINED: 10"

    def test_include(self, tmp_path):
        inc_file = tmp_path / "inc.mod"
        inc_file.write_text("var Y;\n", encoding="utf-8")
        main_file = tmp_path / "main.mod"
        main_file.write_text(f'@#include "{inc_file.name}"\nvar C;\n', encoding="utf-8")

        out = expand_macro(str(main_file), include_paths=[str(tmp_path)])
        lines = out.strip().splitlines()
        assert lines == ["var Y;", "var C;"]

    def test_error_directive(self):
        txt = '@#define err = true\n@#if err\n@#error "Stop execution"\n@#endif'
        with pytest.raises(MacroEvaluationError, match="Stop execution"):
            expand_macro(txt)


class TestDynareDifferentialVerification:
    """Verifies that our macro processor produces 1-to-1 exact matching output against dynare-preprocessor."""

    @pytest.mark.skipif(
        PREPROCESSOR_BIN is None, reason="dynare-preprocessor not available"
    )
    @pytest.mark.parametrize(
        "modfile",
        [
            "examples/dynare/macroprocessor/bkk_1992.mod",
            "examples/modfiles/bkk.mod",
            "examples/modfiles/example1_reporting.mod",
            "examples/modfiles/agtrend.mod",
            "examples/modfiles/Ramsey_Example.mod",
            "examples/dynare/optimal_policy/nk_ramsey_osr.mod",
        ],
    )
    def test_differential_match(self, modfile, tmp_path):
        if not os.path.exists(modfile):
            pytest.skip(f"File {modfile} does not exist")

        gt_output = run_dynare_savemacro(modfile, tmp_path)
        py_output = expand_macro(modfile)

        assert py_output == gt_output


class TestMacroDirectivesPreservationAndDynoModel:
    def test_no_directive_preserved_when_only_if_needed(self):
        txt = (
            "var c k;\n"
            "varexo e;\n"
            "parameters beta;\n\n"
            "model;\n"
            "c = beta*c(+1);\n"
            "end;\n"
        )
        # With only_if_needed=True, text is returned exactly unchanged
        out = expand_macro(txt, only_if_needed=True)
        assert out == txt

    def test_dynomodel_preprocess_option(self):
        from dyno import DynoModel

        mod_txt = (
            "@#define USE_SHOCK = 1\n"
            "var c k;\n"
            "parameters beta;\n"
            "@#if USE_SHOCK\n"
            "varexo e;\n"
            "@#endif\n"
            "model;\n"
            "c = beta*c(+1);\n"
            "k = 1;\n"
            "end;\n"
        )
        # Without preprocess=False, LModFile fails with parser error on @#
        with pytest.raises(Exception):
            DynoModel(filename="test.mod", txt=mod_txt, preprocess=False)

        # By default preprocess=True, macro is expanded and DynoModel parses successfully
        model = DynoModel(filename="test.mod", txt=mod_txt)
        assert "c" in model.symbols["variables"]
        assert "k" in model.symbols["variables"]
        assert "e" in model.symbols["exogenous"]
