# Hierarchy of Classes (clean)

```mermaid
classDiagram
  %% Abstract/base
  class DynoModel

  %% Concrete model types
  class SymbolicModel
  class DynareModel
  class YAMLFile

  %% Parsers / files
  class SymbolicFile
  class DynoFile
  class LModFile

  %% Inheritance
  Model <|-- SymbolicModel
  Model <|-- DynareModel
  Model <|-- YAMLFile

  SymbolicFile <|-- DynoFile
  SymbolicFile <|-- LModFile

  %% Associations
  SymbolicModel --> DynoFile : data
  SymbolicModel --> LModFile : data
  DynareModel --> DynareModel : "uses Dynare preprocessor"
  Model o-- ProductNormal : processes
  Model --> RecursiveSolution : "solve() returns"
  RecursiveSolution ..> Model : model

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
