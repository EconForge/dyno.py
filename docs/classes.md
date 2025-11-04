# Hierarchy of Classes (clean)

```mermaid
classDiagram
  %% Abstract/base
  class DynoModel

  DynoModel: symbols
  DynoModel: context

    SymbolicModel: compute_residuals()
    SymbolicModel: compute_jacobians()
    SymbolicModel: compute_derivatives()

  %% Concrete model types
  class SymbolicModel

class DynareModel

    DynareModel: compute_residuals()
    DynareModel: compute_jacobians()
    DynareModel: compute_derivatives()


  class DynareModel
  class YAMLFile

  %% Parsers / files
  class SymbolicFile

  DynoFile: tree
  DynoFile: FormulaEvaluator evaluator

  LModFile: InterpretModfile evaluator


  class DynoFile
  class LModFile

  %% Inheritance
  DynoModel <|-- SymbolicModel
  DynoModel <|-- DynareModel
  DynoModel <|-- YAMLFile

  SymbolicFile <|-- DynoFile
  SymbolicFile <|-- LModFile

  SymbolicFile: String filename
  SymbolicFile: Tree content
  SymbolicFile: Dict context

  %% Associations
  SymbolicModel --> SymbolicFile : data
  DynareModel --> DynarePreprocessor : data

```

## Legend

- <|-- : Inheritance (subclass)
- o--  : Composition / has-a
- -->  : Association / uses
- ..>  : Reference

Notes:

- `DynareModel` is implemented in `src/dyno/modfile.py` and `modfile_lark.py` and wraps Dynare's preprocessor.
- `SymbolicModel` loads `.dyno` / `.mod` files via `DynoFile` / `LModFile` (subclasses of `SymbolicFile`).
- `YAMLFile` is an alternate `Model` implementation for YAML-described models.
- `Model.processes` may hold a `ProductNormal` (exogenous process).
