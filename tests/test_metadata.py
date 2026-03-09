from dyno import DynoModel
from dyno.larkfiles import DynoFile


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
    assert symbolic.context["metadata"] == symbolic.metadata


def test_dynomodel_exposes_metadata_in_context():
    txt = """
@name: TinyModel
alpha := 0.9
x[~] := 1
x[t] = alpha
"""

    model = DynoModel(txt=txt)

    assert model.context["metadata"]["name"] == "TinyModel"
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
