from pathlib import Path

import pytest

from dyno import DynoModel
from dyno.larkfiles import DynoFile
from dyno.errors import ParserError
from dyno.dynspec.analyze import DefinitionError


def test_dyno_metadata_statements_are_parsed():
    txt = """
@name: RBC
@version: 1
@deterministic: true
@title: "RBC baseline"
a := 1
k[~] := 1
k[t] = a
"""

    symbolic = DynoFile(txt)

    assert symbolic.metadata["name"] == "RBC"
    assert symbolic.metadata["version"] == 1
    assert symbolic.metadata["deterministic"] is True
    assert symbolic.metadata["title"] == "RBC baseline"
    assert "metadata" not in symbolic.context


def test_dynomodel_exposes_metadata_in_context():
    txt = """
@name: TinyModel
alpha := 0.9
x[~] := 1
x[t] = alpha
"""

    model = DynoModel(txt=txt)

    assert "metadata" not in model.context
    assert model.metadata["name"] == "TinyModel"


def test_dynomodel_yaml_argument_parses_model_block():
    txt = """
name: [1, 2, 3]
model: |
    a := 0.1
    e[t] := N(0, 1)
    x[t] = 0.9 * x[t-1]
"""

    model = DynoModel(yaml=txt)

    assert model.metadata["name"] == [1, 2, 3]
    assert "x" in model.symbols["variables"]


def test_dynomodel_yaml_file_parses_model_block(tmp_path):
    p = tmp_path / "wrapped_model.yaml"
    p.write_text(
        """
name: Demo
model: |
    a := 0.1
    e[t] := N(0, 1)
    x[t] = 0.9 * x[t-1]
""",
        encoding="utf-8",
    )

    model = DynoModel(str(p))

    assert model.metadata["name"] == "Demo"
    assert "x" in model.symbols["variables"]


def test_inline_metadata_is_attached_to_equations():
    txt = """
alpha := 0.3
k[~] := 1
y[t] = alpha * k[t-1] :: [production, source=paper]
"""

    symbolic = DynoFile(txt)

    assert len(symbolic.equations) == 1
    eq_meta = symbolic.equations[0].meta.statement_metadata
    assert set(eq_meta["tags"]) == {"production"}
    assert eq_meta["source"] == "paper"


def test_inline_coloncolon_metadata_tags_are_attached_to_equations():
    txt = """
alpha := 0.3
k[~] := 1
y[t] = alpha * k[t-1] :: production, loglinear
"""

    symbolic = DynoFile(txt)

    assert len(symbolic.equations) == 1
    eq_meta = symbolic.equations[0].meta.statement_metadata
    assert set(eq_meta["tags"]) == {"production", "loglinear"}


def test_inline_coloncolon_string_desugars_to_tag():
    txt = """
alpha := 0.3
k[~] := 1
y[t] = alpha * k[t-1] :: "Production function"
"""

    symbolic = DynoFile(txt)

    assert len(symbolic.equations) == 1
    eq_meta = symbolic.equations[0].meta.statement_metadata
    assert eq_meta.get("label") == "Production function"
    assert "tags" not in eq_meta


def test_inline_coloncolon_bracketed_quoted_string_becomes_tag():
    txt = """
alpha := 0.3
k[~] := 1
y[t] = alpha * k[t-1] :: ["transition"]
"""

    symbolic = DynoFile(txt)

    assert len(symbolic.equations) == 1
    eq_meta = symbolic.equations[0].meta.statement_metadata
    assert eq_meta["tags"] == ["transition"]


def test_inline_coloncolon_canonical_list_desugars_to_metadata():
    txt = """
alpha := 0.3
k[~] := 1
y[t] = alpha * k[t-1] :: [production, block=firms]
"""

    symbolic = DynoFile(txt)

    assert len(symbolic.equations) == 1
    eq_meta = symbolic.equations[0].meta.statement_metadata
    assert set(eq_meta["tags"]) == {"production"}
    assert eq_meta["block"] == "firms"


def test_block_metadata_inherits_and_merges_into_statements():
    txt = """
alpha := 0.3
k[~] := 1
[production, block=firms] :: {
    y[t] = alpha * k[t-1]
    [loglinear] :: {
        y[t] = alpha * k[t-1] :: [equation, block=inner]
    }
}
"""

    symbolic = DynoFile(txt)

    assert len(symbolic.equations) == 2

    outer_meta = symbolic.equations[0].meta.statement_metadata
    assert set(outer_meta["tags"]) == {"production"}
    assert outer_meta["block"] == "firms"

    inner_meta = symbolic.equations[1].meta.statement_metadata
    assert set(inner_meta["tags"]) == {"production", "loglinear", "equation"}
    assert inner_meta["block"] == "inner"


def test_floating_metadata_is_rejected():
    txt = """
[production]
y[t] = 1
"""
    with pytest.raises(ParserError):
        DynoFile(txt)


def test_floating_coloncolon_metadata_is_rejected():
    txt = """
:: production
y[t] = 1
"""
    with pytest.raises(ParserError):
        DynoFile(txt)


def test_coloncolon_prefix_block_metadata_is_rejected():
    txt = """
:: [production] {
    y[t] = 1
}
"""
    with pytest.raises(ParserError):
        DynoFile(txt)


def test_print_equations_with_tags(capsys):
    txt = """
alpha := 0.3
k[~] := 1
y[t] = alpha * k[t-1] :: [production]
z[t] = y[t]
"""

    model = DynoModel(txt=txt)

    model.print_equations_with_tags()
    out = capsys.readouterr().out

    assert "1." in out
    assert "2." in out
    assert "y[t]" in out
    assert "alpha" in out
    assert "k[t-1]" in out
    assert "z[t] = y[t]" in out
    assert "[tags: production]" in out
    assert "[tags: -]" in out


def test_filter_equations_by_label():
    model = DynoModel("examples/rbc.dyno")

    matches = model.symbolic.filter_equations(
        lambda eq: eq.metadata.get("label") == "Labor Supply"
    )

    assert len(matches) == 1
    assert matches[0].metadata["label"] == "Labor Supply"
    assert "theta" in matches[0].text


def test_unclosed_metadata_bracket_is_rejected():
    txt = """
y[t] = 1 :: [production
"""
    with pytest.raises(ParserError):
        DynoFile(txt)


def test_empty_metadata_bracket_yields_no_metadata():
    txt = """
alpha := 0.3
k[~] := 1
y[t] = alpha * k[t-1] :: []
"""

    symbolic = DynoFile(txt)

    assert symbolic.equations[0].meta.statement_metadata == {}


def test_metadata_entries_mix_in_any_order():
    txt = """
alpha := 0.3
k[~] := 1
y[t] = alpha * k[t-1] :: [a, id=res, "note", k=2, v="x y", b]
"""

    symbolic = DynoFile(txt)

    eq_meta = symbolic.equations[0].meta.statement_metadata
    assert eq_meta["tags"] == ["a", "note", "b"]
    assert eq_meta["id"] == "res"
    assert eq_meta["k"] == 2
    assert eq_meta["v"] == "x y"


def test_junk_bracket_interior_is_rejected():
    txt = """
y[t] = 1 :: [a + b]
"""
    with pytest.raises(ParserError):
        DynoFile(txt)


def test_junk_block_bracket_is_rejected():
    txt = """
[the transition block] :: {
    y[t] = 1
}
"""
    with pytest.raises(ParserError):
        DynoFile(txt)


def test_numeric_metadata_values_are_numbers():
    txt = """
y[t] = 1 :: [k=-2, x=1.5e3, z=+0.5]
"""

    symbolic = DynoFile(txt)

    assert symbolic.equations[0].meta.statement_metadata == {
        "k": -2,
        "x": 1500.0,
        "z": 0.5,
    }


def test_duplicate_metadata_key_is_rejected():
    txt = """
y[t] = 1 :: [id=a, id=b]
"""
    with pytest.raises(DefinitionError, match="Duplicate metadata key: id"):
        DynoFile(txt)


def test_invalid_coloncolon_text_reports_position():
    txt = """
y[t] = 1 :: id=res
"""
    with pytest.raises(DefinitionError) as exc:
        DynoFile(txt)
    assert str(exc.value).startswith("(2, ")


def test_inline_bracket_without_coloncolon_is_rejected():
    txt = """
alpha := 0.3
y[t] = alpha [production, id=res]
"""
    with pytest.raises(ParserError, match="write `<statement> :: \\[tags\\]`"):
        DynoFile(txt)


def test_block_tag_without_coloncolon_is_rejected():
    txt = """
[production] {
    y[t] = 1
}
"""
    with pytest.raises(ParserError, match="write `\\[tags\\] :: \\{"):
        DynoFile(txt)


def test_space_before_index_is_indexing_not_annotation():
    txt = """
alpha := 0.3
k[~] := 1
y[t] = alpha * k [t-1]
"""

    model = DynoModel(txt=txt)

    assert model.symbolic.equations[0].meta.statement_metadata == {}
    assert "k" in model.symbols["variables"]


@pytest.mark.parametrize(
    "name",
    ["neo.dyno", "rbc.dyno", "ramst.dyno", "neoclassical_ramsey.dyno", "rbc_dolo.dyno"],
)
def test_examples_still_parse(name):
    path = Path(__file__).parent.parent / "examples" / name
    DynoFile(path.read_text())


def test_file_with_every_annotation_position():
    txt = """
@name: Demo

alpha <- 0.3

[block_a, id=g1, note="transition"] :: {
    k[t] = (1-delta)*k[t-1] + i[t]   :: [id=lom, capital]
    y[t] = k[t-1]^alpha              :: "production"
}

c[t] = y[t] - i[t] :: [budget, id=res]
"""

    symbolic = DynoFile(txt)

    metas = [eq.meta.statement_metadata for eq in symbolic.equations]
    assert metas[0] == {
        "tags": ["block_a", "capital"],
        "id": "lom",
        "note": "transition",
    }
    assert metas[1] == {
        "tags": ["block_a"],
        "id": "g1",
        "note": "transition",
        "label": "production",
    }
    assert metas[2] == {"tags": ["budget"], "id": "res"}
