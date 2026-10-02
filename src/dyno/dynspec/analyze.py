from lark.visitors import Transformer, Interpreter
from lark.tree import Tree
from lark.lexer import Token
import math
import yaml
from typing import Dict, Any, Callable, Union, List
from .autodiff import DNumber as DN
from .language import Normal
import math


from dyno.errors import ParserError


class DefinitionError(ParserError):

    def __init__(self, msg, tree=None):

        super().__init__(str(msg))
        self.msg = msg
        self.tree = tree
        meta = getattr(tree, "meta", None)
        if meta is not None and not getattr(meta, "empty", True):
            self.line = getattr(meta, "line", None)
            self.column = getattr(meta, "column", None)

    def __str__(self):

        meta = getattr(self.tree, "meta", None)
        if meta is None or getattr(meta, "empty", True):
            return str(self.msg)
        return f"({meta.line}, {meta.column}): {self.msg}"


def _to_number(text: str) -> Union[int, float]:
    """Convert a numeric literal (e.g. ``-2``, ``1.5e3``) to int or float."""
    try:
        return int(text)
    except ValueError:
        return float(text)


def _normal_distribution(*args: Any) -> Normal:
    if len(args) == 1:
        u = 0.0
        s = args[0]
    elif len(args) == 2:
        u = args[0]
        s = args[1]
    else:
        raise TypeError(
            f"N() takes 1 or 2 arguments: N(std) or N(mean, std), but {len(args)} were given"
        )
    val = getattr(s, "value", s)
    if isinstance(val, (int, float)) and val < 0:
        raise ValueError(f"Standard deviation must be non-negative, got {val}")
    return Normal(Sigma=[[s**2]], Μ=[u])


function_table_0 = {
    "exp": math.exp,
    "log": math.log,
    "sqrt": math.sqrt,
    "abs": math.fabs,
}


class FormulaEvaluator(Interpreter):
    """
    An interpreter that evaluates mathematical formulas as defined by the grammar.

    This class can evaluate:
    - Basic arithmetic operations (add, sub, mul, div, pow, neg)
    - Numbers and symbols (constants, values, variables)
    - Function calls
    - Assignments and equations
    """

    def __init__(
        self,
        context: Dict[str, Any] = {},
        function_table: Dict[str, Callable] = function_table_0,
        unknown_as_nan=True,
        raise_on_nan=False,
        steady_state: bool = False,
    ):
        """
        Initialize the evaluator.

        Args:
            symbol_table: Dictionary mapping symbol names to their values
            function_table: Dictionary mapping function names to callable functions
            steady_state: If True, evaluates variables at their steady state (only the name of the symbol is taken into account)
            raise_on_nan: If True, raise a DefinitionError as soon as any node evaluates to NaN,
                pinpointing the offending subtree instead of letting NaN propagate silently.
        """
        super().__init__()

        self.function_table: Dict[str, Callable[..., Any]] = dict(function_table or {})
        self.unknown_as_nan = unknown_as_nan
        self.raise_on_nan = raise_on_nan
        self.steady_state = steady_state

        self.constants = context.get("constants", {})
        self.processes = context.get("processes", {})
        self.values = context.get("values", {})
        self.variables = context.get("variables", {})
        self.steady_states = context.get("steady_states", {})
        self.metadata = context.get("metadata", {})

        self.time = None  # None or integer
        self.errors: List[Any] = []

        # Add default mathematical functions
        from .autodiff import MATH_FUNCTIONS

        self.function_table.update(MATH_FUNCTIONS)
        self.function_table.update({"N": _normal_distribution})

    def _undefined(self, message: str, tree):
        """Report an undefined value: raise if `unknown_as_nan` is False, else return NaN."""
        if not self.unknown_as_nan:
            raise DefinitionError(message, tree=tree)
        err = DefinitionError(message, tree=tree)
        self.errors.append(err)
        import warnings
        from dyno.errors import UndefinedSymbolWarning

        if getattr(tree, "data", None) == "constant":
            try:
                loc = f" at ({tree.meta.line}, {tree.meta.column})"
            except Exception:
                loc = ""
            warnings.warn(
                f"{message}{loc}. Defaulting to NaN.",
                UndefinedSymbolWarning,
                stacklevel=3,
            )
        return math.nan

    def visit(self, tree):
        """Visit a node, optionally raising as soon as its result is NaN.

        Centralizing the check here (rather than in every node evaluator) means it
        applies uniformly and reports the innermost subtree where the NaN first appears.
        """
        result = super().visit(tree)
        if self.raise_on_nan and isinstance(result, float) and math.isnan(result):
            raise DefinitionError(
                f"NaN encountered while evaluating `{tree.data}`", tree=tree
            )
        return result

    # Arithmetic operations
    def add(self, tree):
        """Handle addition: a + b"""
        left = self.visit(tree.children[0])
        right = self.visit(tree.children[1])
        return left + right

    def sub(self, tree):
        """Handle subtraction: a - b"""
        left = self.visit(tree.children[0])
        right = self.visit(tree.children[1])
        return left - right

    def mul(self, tree):
        """Handle multiplication: a * b"""
        left = self.visit(tree.children[0])
        right = self.visit(tree.children[1])
        return left * right

    def div(self, tree):
        """Handle division: a / b"""
        left = self.visit(tree.children[0])
        right = self.visit(tree.children[1])
        # TODO: maybe add a safe division?
        # if right == 0:
        #     raise ZeroDivisionError("Division by zero")
        return left / right

    def pow(self, tree):
        """Handle exponentiation: a ^ b or a ** b"""
        base = self.visit(tree.children[0])
        exponent = self.visit(tree.children[1])
        return base**exponent

    def neg(self, tree):
        """Handle negation: -a"""
        value = self.visit(tree.children[0])
        return -value

    # Numbers and literals
    def number(self, tree):
        """Handle numeric literals"""
        value = tree.children[0].value
        # Try to parse as int first, then float
        try:
            return int(value)
        except ValueError:
            return float(value)

    # Symbols
    def constant(self, tree):
        """Handle constants (symbols without time indexing)"""
        name = str(tree.children[0].children[0])
        if name in self.constants:
            return self.constants[name]
        else:
            return self._undefined(f"Undefined value: {name}", tree)

    def value(self, tree):
        """Handle values with specific time: name[time]"""
        name = str(tree.children[0].children[0])
        time = int(tree.children[1].children[0])

        # Create a key for the symbol table
        if self.steady_state:
            if name not in self.steady_states:
                return self._undefined(
                    f"Undefined steady state for value {name}[~]", tree
                )
            return self.steady_states[name]
        else:
            if name not in self.values:
                return self._undefined(f"Undefined value {name}[~]", tree)
            else:
                vvs = self.values[name]
                if time not in vvs:
                    return self._undefined(f"Undefined value {name}[{time}]", tree)
                else:
                    return vvs[time]

    def variable(self, tree):
        """Handle variables with time indexing: name[t+shift]"""
        name = str(tree.children[0].children[0])
        index = str(tree.children[1].children[0])  # Usually 't'
        shift = int(tree.children[2].children[0])

        # TODO deal with index ~ #### this should disappear
        if name not in self.variables:
            self.variables[name] = {}

        if self.time is not None:
            time = self.time + shift
            vvs = self.values.get(name, {})
            if time not in vvs:
                return self._undefined(f"Undefined value {name}[{time}]", tree)
            return vvs[time]
        elif self.steady_state or (index == "~"):
            # from rich import print
            if name not in self.steady_states:
                return self._undefined(
                    f"Undefined steady state for variable {name}[~]", tree
                )
            return self.steady_states[name]
        else:
            if shift not in self.variables[name]:
                return self._undefined(f"Undefined variable {name}[{shift}]", tree)
            return self.variables[name][shift]

    # Function calls
    def call(self, tree):
        """Handle function calls: func_name(arg)"""
        func_name = str(tree.children[0].children[0])

        if func_name in self.function_table:
            args = [self.visit(c) for c in tree.children[1:]]
            return self.function_table[func_name](*args)
        elif func_name == "steady_state":
            arg = tree.children[1]
            if getattr(arg, "data", None) == "variable":
                name = str(arg.children[0].children[0])
                if name not in self.steady_states:
                    return self._undefined(
                        f"Undefined steady state for variable {name}[~]", tree
                    )
                return self.steady_states[name]
            else:
                prev_ss = self.steady_state
                try:
                    self.steady_state = True
                    return self.visit(arg)
                finally:
                    self.steady_state = prev_ss
        else:
            raise ValueError(f"Undefined function: {func_name}")

    def bare_formula(self, tree):
        """Handle a standalone formula equation (implicitly `formula = 0`)"""
        return self.visit(tree.children[0])


class AssignmentEvaluator(FormulaEvaluator):

    def __init__(
        self,
        context: Dict[str, Any] = {},
        symbol_table: Dict[str, Any] = {},
        function_table: Dict[str, Callable] = {},
        unknown_as_nan=True,
        raise_on_nan=False,
        calibration: Dict[str, Any] = {},
    ):
        """
        Initialize the evaluator.

        Args:
            symbol_table: Dictionary mapping symbol names to their values
            function_table: Dictionary mapping function names to callable functions
            steady_state: If True, evaluates variables at their steady state (only the name of the symbol is taken into account)
            calibration: Values that override constants defined in the model
            raise_on_nan: If True, raise a DefinitionError as soon as any node evaluates to NaN.
        """
        super().__init__()

        self.function_table: Dict[str, Callable[..., Any]] = dict(function_table or {})
        self.unknown_as_nan = unknown_as_nan
        self.raise_on_nan = raise_on_nan

        self.__calibration__ = calibration.copy()

        self.constants = context.get("constants", self.__calibration__.copy())
        self.processes = context.get("processes", {})
        self.values = context.get("values", {})
        self.variables = context.get("variables", {})
        self.steady_states = context.get("steady_states", {})
        self.metadata = context.get("metadata", {}).copy()

        self.equations: List[Any] = []
        self.equation_metadata: List[Dict[str, Any]] = []
        self.block_metadata_entries: List[Any] = []
        self._metadata_stack: List[Dict[str, Any]] = [{"tags": []}]
        self.time = None  # None or integer
        self.errors: List[Any] = []

        # Add default mathematical functions
        from .autodiff import MATH_FUNCTIONS

        self.function_table.update(MATH_FUNCTIONS)
        self.function_table.update({"N": _normal_distribution})

    def _normalize_metadata(self, item_list: List[tuple[str, Any]]) -> Dict[str, Any]:
        tags: List[str] = []
        kv: Dict[str, Any] = {}
        for kind, value in item_list:
            if kind == "tag":
                if value not in tags:
                    tags.append(value)
            elif kind == "kv":
                key, val = value
                kv[key] = val
        if len(tags) > 0:
            kv["tags"] = tags
        return kv

    def _merge_metadata(
        self, base: Dict[str, Any], override: Dict[str, Any]
    ) -> Dict[str, Any]:
        merged = dict(base)

        base_tags = list(base.get("tags", []))
        override_tags = list(override.get("tags", []))
        tags = list(base_tags)
        for tag in override_tags:
            if tag not in tags:
                tags.append(tag)
        if len(tags) > 0:
            merged["tags"] = tags
        elif "tags" in merged:
            merged.pop("tags")

        for key, value in override.items():
            if key == "tags":
                continue
            merged[key] = value

        return merged

    def _attach_statement_metadata(self, node: Tree, metadata: Dict[str, Any]) -> None:
        try:
            setattr(node.meta, "statement_metadata", metadata)
        except Exception:
            pass

    def _split_metadata_items(self, body: str) -> List[str]:
        items: List[str] = []
        current: List[str] = []
        in_single = False
        in_double = False

        for char in body:
            if char == "'" and not in_double:
                in_single = not in_single
            elif char == '"' and not in_single:
                in_double = not in_double

            if char == "," and not in_single and not in_double:
                item = "".join(current).strip()
                if len(item) > 0:
                    items.append(item)
                current = []
                continue

            current.append(char)

        tail = "".join(current).strip()
        if len(tail) > 0:
            items.append(tail)
        return items

    def _coerce_metadata_value(self, raw_value: str) -> Any:
        stripped = raw_value.strip()
        parsed = yaml.safe_load(stripped)
        if isinstance(parsed, (int, float, str, bool)):
            return parsed
        if parsed is None and stripped in ("", "null", "~"):
            return parsed
        if parsed is not None:
            return str(parsed)
        return stripped

    def _annotation_to_metadata(self, ann_tree: Tree) -> Dict[str, Any]:
        """Convert a parsed annotation node to a metadata dict.

        The grammar has already separated the entries (kv, baretag,
        strtag): a quoted string in bracket position is a tag, numbers
        become int/float, and other kv values receive YAML coercion.
        """
        normalized_items: List[tuple[str, Any]] = []
        seen_keys: set[str] = set()
        for entry in ann_tree.children:
            if entry.data == "kv":
                key = str(entry.children[0].children[0])
                if key in seen_keys:
                    raise DefinitionError(f"Duplicate metadata key: {key}", ann_tree)
                seen_keys.add(key)
                valnode = entry.children[1]
                value: Any
                if isinstance(valnode, Tree):  # cname -> name
                    value = self._coerce_metadata_value(str(valnode.children[0]))
                elif valnode.type == "SIGNED_NUMBER":
                    value = _to_number(str(valnode))
                else:  # quoted string
                    value = self._coerce_metadata_value(str(valnode))
                normalized_items.append(("kv", (key, value)))
            elif entry.data == "baretag":
                normalized_items.append(("tag", str(entry.children[0].children[0])))
            else:  # strtag: a quoted string in a bracket is a tag
                value = self._coerce_metadata_value(str(entry.children[0]))
                if not isinstance(value, str):
                    raise DefinitionError(
                        f"Invalid metadata tag: {entry.children[0]}", ann_tree
                    )
                normalized_items.append(("tag", value))
        if not normalized_items:
            return {"tags": []}
        return self._normalize_metadata(normalized_items)

    def statement_metadata(self, tree):
        """Handler for statement_metadata nodes.

        The node's single child is either an annotation subtree (a
        parsed `:: [entries]` bracket) or a META_TEXT token (free text
        after ::, which never begins with a bracket).
        """
        child = tree.children[0]
        if isinstance(child, Tree):
            return self._annotation_to_metadata(child)
        # Bare text that came after :: — treat as tag(s) / quoted label
        try:
            return self._parse_inline_content(str(child).strip())
        except DefinitionError as e:
            if e.tree is None:
                e.tree = tree
            raise

    def _parse_inline_content(self, content: str) -> "Dict[str, Any]":
        """Parse the text content that appears after :: (no :: prefix expected)."""
        if not content:
            raise DefinitionError("Empty :: metadata")

        if content[0] in ('"', "'"):
            if len(content) < 2 or content[-1] != content[0]:
                raise DefinitionError("Malformed :: metadata string")
            value = self._coerce_metadata_value(content)
            if not isinstance(value, str):
                raise DefinitionError("Invalid :: metadata string")
            return self._normalize_metadata([("kv", ("label", value))])

        items = self._split_metadata_items(content)
        if not items:
            raise DefinitionError("Invalid :: metadata usage")
        normalized: "List[tuple[str, Any]]" = []
        for item in items:
            candidate = item.strip()
            if "=" in candidate:
                raise DefinitionError(":: metadata only accepts tags or quoted string")
            if not candidate.isidentifier():
                raise DefinitionError(f"Invalid :: metadata tag: {candidate}")
            normalized.append(("tag", candidate))
        return self._normalize_metadata(normalized)

    def assignment(self, tree):
        """Handle assignments: symbol := value or symbol <- value"""
        symbol_tree = tree.children[0]
        value = self.visit(tree.children[1])

        name = str(symbol_tree.children[0].children[0])

        if symbol_tree.data == "constant":

            if name in self.__calibration__:
                # print(f"Warning: constant {name} calibrated to {self.__calibration__[name]}; assignment ignored.")
                return
            if name in self.constants:
                import warnings

                from dyno.errors import RedefinitionWarning

                meta = getattr(symbol_tree, "meta", None)
                where = (
                    f" (line {meta.line})"
                    if meta is not None and not meta.empty
                    else ""
                )
                warnings.warn(
                    f"Constant {name} redefined{where}; keeping its first value.",
                    RedefinitionWarning,
                    stacklevel=2,
                )
            else:
                self.constants[name] = value
            # self.symbol_table[key] = value

        elif symbol_tree.data == "value":
            if name not in self.values:
                self.values[name] = {}
            time = int(symbol_tree.children[1].children[0])
            self.values[name][time] = value

        elif symbol_tree.data == "variable":
            index = str(symbol_tree.children[1].children[0])
            shift = int(symbol_tree.children[2].children[0])

            if index == "~":
                key = f"{name}[~]"
                # self.symbol_table[key] = value
                if name not in self.steady_states:
                    self.steady_states[name] = value
            else:
                assert shift == 0
                if name in self.processes:
                    raise Exception(f"Warning: invalid redefinition of process {name}.")
                else:

                    # TODO: check that value is a process
                    self.processes[(name,)] = value
                    self.steady_states[name] = float(value.Μ[0])

        return value

    def quantified_assignment(self, tree):

        bounds = tree.children[0]
        assert bounds.data == "t_double_bound"
        lower = self.visit(bounds.children[0])
        upper = self.visit(bounds.children[1])
        if not (isinstance(lower, int) and isinstance(upper, int) and lower < upper):
            raise ValueError(
                f"Invalid bounds in quantified assignment: {lower}, {upper}"
            )
        dates = range(lower, upper)

        symbol_tree = tree.children[1]
        name = str(symbol_tree.children[0].children[0])

        if name not in self.values:
            self.values[name] = {}

        for d in dates:

            self.time = d
            self.constants["t"] = d

            value = self.visit(tree.children[2])

            name = str(symbol_tree.children[0].children[0])
            index = str(symbol_tree.children[1].children[0])
            shift = int(symbol_tree.children[2].children[0])
            assert index == "t" and shift == 0

            # self.symbol_table[key] = value
            self.values[name][d] = value

        # self.time = original_time
        self.constants.pop("t", None)
        self.time = None

    def metadata_scalar(self, tree):
        # Parse scalar values with YAML semantics (numbers, booleans, quoted strings).
        raw_value = str(tree.children[0]).strip()
        try:
            return yaml.safe_load(raw_value)
        except yaml.YAMLError:
            return raw_value

    # Block handling
    def annotated_statement(self, tree):
        statement = tree.children[0]
        meta = {"tags": []}
        if len(tree.children) > 1:
            meta = self.visit(tree.children[1])  # statement_metadata node

        inherited = self._metadata_stack[-1]
        merged = self._merge_metadata(inherited, meta)

        if statement.data in ("equality", "bare_formula", "formula"):
            self._attach_statement_metadata(statement, merged)
            self.equations.append(statement)
            self.equation_metadata.append(merged)
            return statement

        self._attach_statement_metadata(statement, merged)
        return self.visit(statement)

    def block(self, tree):
        for child in tree.children:
            self.visit(child)
        return None

    def block_tag(self, tree):
        """Handler for block_tag nodes: a parsed annotation bracket."""
        return self._annotation_to_metadata(tree.children[0])

    def annotated_block(self, tree):
        block_meta = self.visit(tree.children[0])  # block_tag node
        inherited = self._metadata_stack[-1]
        merged = self._merge_metadata(inherited, block_meta)
        self.block_metadata_entries.append(merged)
        self._metadata_stack.append(merged)
        try:
            self.visit(tree.children[1])  # block node
        finally:
            self._metadata_stack.pop()
        return None

    def model_metadata(self, tree):
        """Handle top-level @key: value declarations."""
        key = str(tree.children[0])  # NAME token
        raw = str(tree.children[1])  # METADATA_SCALAR token
        raw_stripped = raw.strip()
        mute = False
        if raw_stripped.endswith(";"):
            mute = True
            raw_stripped = raw_stripped[:-1].rstrip()
        if key == "run":
            if ";" in raw_stripped:
                raise DefinitionError(
                    f"Invalid @run command syntax: unexpected ';' in '{raw.strip()}'. Semicolons are only permitted at the end of the command line.",
                    tree=tree,
                )
            try:
                import yaml

                value = yaml.safe_load(raw_stripped)
            except Exception as e:
                raise DefinitionError(
                    f"Invalid @run command YAML syntax in '{raw.strip()}': {e}",
                    tree=tree,
                ) from e

            # Validate run command structure
            if isinstance(value, str):
                if not value.isidentifier():
                    raise DefinitionError(
                        f"Invalid @run command name: '{value}' is not a valid identifier.",
                        tree=tree,
                    )
                if mute:
                    value = {"command": value, "options": {}, "mute": True}
            elif isinstance(value, dict):
                if "command" in value:
                    cmd_val = value.get("command")
                    if not isinstance(cmd_val, str) or not cmd_val.isidentifier():
                        raise DefinitionError(
                            f"Invalid @run command: 'command' must be a valid identifier string, got {cmd_val!r}.",
                            tree=tree,
                        )
                    if "options" in value and not isinstance(value["options"], dict):
                        raise DefinitionError(
                            f"Invalid @run command: 'options' must be a dictionary, got {type(value['options']).__name__}.",
                            tree=tree,
                        )
                    value = dict(value)
                    if mute:
                        value["mute"] = True
                else:
                    non_mute_keys = [k for k in value if k != "mute"]
                    if len(non_mute_keys) != 1:
                        raise DefinitionError(
                            f"Invalid @run command mapping: expected a single command name key, got {list(value.keys())}.",
                            tree=tree,
                        )
                    cmd_k = non_mute_keys[0]
                    if not isinstance(cmd_k, str) or not cmd_k.isidentifier():
                        raise DefinitionError(
                            f"Invalid @run command name: '{cmd_k}' is not a valid identifier.",
                            tree=tree,
                        )
                    opts = value[cmd_k]
                    if opts is not None and not isinstance(opts, dict):
                        raise DefinitionError(
                            f"Invalid options for @run command '{cmd_k}': options must be a dictionary or null, got {type(opts).__name__}.",
                            tree=tree,
                        )
                    if mute:
                        value = {"command": cmd_k, "options": opts or {}, "mute": True}
            else:
                raise DefinitionError(
                    f"Invalid @run directive: command must be a name or a mapping, got {type(value).__name__}.",
                    tree=tree,
                )

            if key in self.metadata:
                current = self.metadata[key]
                if isinstance(current, list):
                    current.append(value)
                else:
                    self.metadata[key] = [current, value]
            else:
                self.metadata[key] = value
        else:
            try:
                import yaml

                value = yaml.safe_load(raw_stripped)
            except Exception:
                value = raw_stripped
            self.metadata[key] = value
            if mute:
                self.metadata[f"_muted_{key}"] = True
        return value

    def assignment_block(self, tree):
        """Handle a block of assignments"""
        results = []
        for child in tree.children:
            if hasattr(child, "data"):  # Skip newlines
                result = self.visit(child)
                results.append(result)
        return results

    # equations are just stored separately (without evaluation)
    def equation_block(self, tree):
        """Handle a block of equations"""
        results = []
        for child in tree.children:
            if hasattr(child, "data"):  # Skip newlines
                result = self.visit(child)
                results.append(result)
        return results

    def free_block(self, tree):
        """Handle a mixed block of equations and assignments"""
        for child in tree.children:
            if hasattr(child, "data"):
                self.visit(child)
        return []


import math

function_table_0 = {
    "exp": math.exp,
    "log": math.log,
    "sqrt": math.sqrt,
    "abs": math.fabs,
}


class EquationsEvaluator(FormulaEvaluator):

    def __init__(
        self,
        context: Dict[str, Any] = {},
        function_table: Dict[str, Callable] = function_table_0,
        steady_state=False,
        diff=False,
        unknown_as_nan=True,
        raise_on_nan=False,
        assign=False,
    ):
        """
        Initialize the evaluator.

        Args:
            symbol_table: Dictionary mapping symbol names to their values
            function_table: Dictionary mapping function names to callable functions
            steady_state: If True, evaluates variables at their steady state (only the name of the symbol is taken into account)
            assign: If True, an equality `x[t] = <rhs>` stores the evaluated right-hand
                side as the current value of `x[t]` instead of returning a residual, so
                that subsequent equations can refer to the updated value.
            raise_on_nan: If True, raise a DefinitionError as soon as any node evaluates to NaN.
        """
        super().__init__()
        # self.symbol_table = symbol_table or {}

        self.function_table: Dict[str, Callable[..., Any]] = dict(function_table or {})
        self.steady_state = steady_state
        self.diff = diff
        self.unknown_as_nan = unknown_as_nan
        self.raise_on_nan = raise_on_nan
        self.assign = assign

        self.constants = context.get("constants", {})
        self.processes = context.get("processes", {})
        self.values = context.get("values", {})
        self.variables = context.get("variables", {})

        self.steady_states = context.get("steady_states", {})

        self.equations: List[Any] = []
        self.time = None  # None or integer
        self.errors: List[Any] = []

        # Add default mathematical functions
        from .autodiff import MATH_FUNCTIONS

        self.function_table.update(MATH_FUNCTIONS)
        # self.function_table.update({"N": (lambda u, v: Normal(Sigma=[[v]], Μ=[u]))})

    def evaluate(self, equations):
        """Evaluate a sequence of equation trees in order, returning their results.

        When `assign=True`, each equality's right-hand side is evaluated using the
        values assigned by previously evaluated equations in this same call.
        """
        return [self.visit(eq) for eq in equations]

    def _assign_variable(self, var_tree, value):
        """Store `value` as the current value of the variable referenced by var_tree."""
        if not (hasattr(var_tree, "data") and var_tree.data == "variable"):
            raise DefinitionError(
                "Left-hand side of an assignment must be a variable", tree=var_tree
            )
        name = str(var_tree.children[0].children[0])
        shift = int(var_tree.children[2].children[0])
        if self.time is not None:
            self.values.setdefault(name, {})[self.time + shift] = value
        else:
            self.variables.setdefault(name, {})[shift] = value

    # Equations and assignments
    def equality(self, tree):
        """Handle equations: left = right.

        Returns the residual (right - left) by default. If `assign` is set, the
        right-hand side is instead stored as the value of the left-hand side
        variable and returned as-is.
        """
        right = self.visit(tree.children[1])
        if self.assign:
            self._assign_variable(tree.children[0], right)
            return right
        left = self.visit(tree.children[0])
        return right - left  # Return difference for equation solving


# class EvalEquations(FormulaEvaluator):

#     # Equations and assignments
#     def equality(self, tree):
#         """Handle equations: left = right. Returns the difference (should be 0 for equality)"""
#         left = self.visit(tree.children[0])
#         right = self.visit(tree.children[1])
#         return right - left  # Return difference for equation solving
