# Hierarchy of Classes (clean)

```mermaid
classDiagram
  %% Abstract/base
  class AbstractModel

  %% Concrete model types
  class DynoModel
  class DynareModel
  class YAMLFile

  %% Parsers / files
  class SymbolicFile
  class DynoFile
  class LModFile

  %% Inheritance
  Model <|-- DynoModel
  Model <|-- DynareModel
  Model <|-- YAMLFile

  SymbolicFile <|-- DynoFile
  SymbolicFile <|-- LModFile

  %% Associations
  DynoModel --> DynoFile : data
  DynoModel --> LModFile : data
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
- `DynoModel` loads `.dyno` / `.mod` files via `DynoFile` / `LModFile` (subclasses of `SymbolicFile`).
- `YAMLFile` is an alternate `Model` implementation for YAML-described models.
- `Model.processes` may hold a `ProductNormal` (exogenous process).
