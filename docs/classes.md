# Hierarchy of Model Classes

```mermaid
classDiagram
    direction TB
    %% Core model inheritance
    class AbstractModel
    class DynoModel
    class DynareModel
    class YAMLFile
    class DynoFile
    class LModFile

    <<abstract>> AbstractModel

    AbstractModel <|-- DynoModel
    AbstractModel <|-- DynareModel
    
    DynoModel ..> DynoFile : loads .dyno
    DynoModel ..> LModFile : loads .mod (Lark)
    AbstractModel <|-- YAMLFile
```


Notes:

- `AbstractModel` is the abstract base class for models.
- `DynoModel`, `DynareModel`, and `YAMLFile` are the main concrete model types.
- `DynoModel` uses `DynoFile` / `LModFile` (both `SymbolicFile` subclasses) to parse textual model descriptions.
