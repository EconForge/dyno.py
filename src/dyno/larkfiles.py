import math
from dataclasses import dataclass
from lark import Tree, Token
from dyno.dynspec.grammar import parser, str_expression
from dyno.dynspec.analyze import (
    FormulaEvaluator,
    AssignmentEvaluator,
    EquationsEvaluator,
)
from typing_extensions import Self
import numpy as np
from typing import List, Dict, Any
import re
from dyno.errors import LARKParserError, ParserError
from lark.exceptions import UnexpectedInput


@dataclass(frozen=True)
class EquationMatch:
    equation: Tree
    metadata: Dict[str, Any]

    @property
    def text(self) -> str:
        return str_expression(self.equation)

    def __str__(self) -> str:
        return self.text


class SymbolicModel:
    filename: str
    content: str
    tree: Tree
    context: Dict
    equations: List[Tree]
    parameters: List[str]
    evaluator: Any | None
    metadata: Dict[str, Any]
    # declarations: List[Tree]

    def __init__(self, content: str, filename="<none>.dyno") -> None:
        # store provided filename and content on the instance
        self.filename = filename
        self.content = content
        self.parameters = []
        self.evaluator = None
        self.metadata = {}

    def latex_equations(self):

        from dyno.dynspec.latex import latex

        def _latex_text_escape(text: str) -> str:
            return (
                text.replace("\\", r"\textbackslash{}")
                .replace("{", r"\{")
                .replace("}", r"\}")
                .replace("_", r"\_")
                .replace("%", r"\%")
                .replace("$", r"\$")
                .replace("&", r"\&")
                .replace("#", r"\#")
                .replace("^", r"\^")
                .replace("~", r"\~{}")
            )

        def _split_equality(eq_latex: str) -> tuple[str, str | None]:
            parts = eq_latex.split(" = ", 1)
            if len(parts) == 2:
                return parts[0], parts[1]
            return eq_latex, None

        lines: list[str] = []
        for i, eq in enumerate(self.equations, start=1):
            eq_latex = latex(eq)
            meta = getattr(getattr(eq, "meta", None), "statement_metadata", {})
            label_cell = r"\text{}"
            if isinstance(meta, dict) and "label" in meta:
                label = str(meta["label"])
                label = _latex_text_escape(label)
                label_cell = r"\text{" + label + r"}"

            lhs, rhs = _split_equality(eq_latex)
            if rhs is None:
                equation_part = f"{lhs}"
            else:
                equation_part = f"{lhs} = {rhs}"

            # Use displaystyle with explicit spacing, no numbering in LaTeX
            lines.append(f"$$\\displaystyle {label_cell} \\quad {equation_part}$$")

        return "\n".join(lines)

    def equations_table_markdown(self):
        """Return equations formatted as separate display-math blocks."""
        from dyno.dynspec.latex import latex

        def _latex_text_escape(text: str) -> str:
            return (
                text.replace("\\", r"\textbackslash{}")
                .replace("{", r"\{")
                .replace("}", r"\}")
                .replace("_", r"\_")
                .replace("%", r"\%")
                .replace("$", r"\$")
                .replace("&", r"\&")
                .replace("#", r"\#")
                .replace("^", r"\^")
                .replace("~", r"\~{}")
            )

        lines: list[str] = []

        for i, eq in enumerate(self.equations, start=1):
            eq_latex = latex(eq)
            meta = getattr(getattr(eq, "meta", None), "statement_metadata", {})
            label_text = r"\text{}"
            if isinstance(meta, dict) and "label" in meta:
                label_text = str(meta["label"])
                label_text = _latex_text_escape(label_text)
                label_text = r"\text{" + label_text + "}"

            lines.append(
                f"$$\\displaystyle {label_text} \\quad {eq_latex} \\quad ({i})$$"
            )

        return "\n\n".join(lines)

    def eval_residuals(self, context: dict | None = None) -> list:

        if context is None:
            context = self.context

        fe = EquationsEvaluator(context)
        fe.steady_state = True
        residuals = [fe.visit(eq) for eq in self.equations]
        return residuals

    def iter_equations_with_metadata(self):
        """Yield pairs of (equation_tree, metadata_dict) for all equations."""
        for eq in self.equations:
            meta = getattr(getattr(eq, "meta", None), "statement_metadata", {})
            if meta is None:
                meta = {}
            yield eq, meta

    def filter_equations(self, predicate):
        """Return equation entries whose metadata-aware wrapper matches predicate."""
        matches: list[EquationMatch] = []
        for eq, meta in self.iter_equations_with_metadata():
            entry = EquationMatch(equation=eq, metadata=dict(meta))
            if predicate(entry):
                matches.append(entry)
        return matches


class DynoFile(SymbolicModel):
    """Class for .dyno files"""

    def __init__(self, content: str, filename="<none>.dyno") -> None:
        super().__init__(content, filename)
        self.content = content
        self.parse()

    def parse(self: Self) -> None:

        txt = self.content

        try:
            tree = parser.parse(
                txt,
                start="free_block",
            )
        except UnexpectedInput as e:
            raise LARKParserError(e, txt) from e

        self.tree = tree

        self.process_assignments()

        # Separate assignments processing from parsing.

    def process_assignments(self, **calib) -> Tree:

        fe = AssignmentEvaluator(calibration=calib)
        fe.visit(self.tree)
        context = {
            "constants": fe.constants,
            "variables": fe.variables,
            "values": fe.values,
            "processes": fe.processes,
            "steady_states": fe.steady_states,
        }
        self.context = context
        self.metadata = fe.metadata
        self.equations = fe.equations
        self.processes = fe.processes
        self.residuals = self.eval_residuals()


from dyno.dynspec.dynare import (
    ModFileTransformer,
    modfile_grammar,
    InterpretModfile,
)


def substitute_model_local_vars(content: str) -> str:
    """Substitute Dynare model-local variables (`# var = expr;`) inside `model; ... end;` blocks."""
    pattern = re.compile(
        r"(\bmodel\s*(?:\([^)]*\))?\s*;)(.*?)(\bend\s*;)", re.DOTALL | re.IGNORECASE
    )

    def repl_model(match: re.Match) -> str:
        header = match.group(1)
        body = match.group(2)
        footer = match.group(3)

        local_var_pattern = re.compile(
            r"^[ \t]*#[ \t]*([a-zA-Z_]\w*)[ \t]*=[ \t]*(.*?);", re.MULTILINE
        )
        raw_vars: list[tuple[str, str]] = []
        for m in local_var_pattern.finditer(body):
            raw_vars.append((m.group(1), m.group(2).strip()))

        if not raw_vars:
            return match.group(0)

        new_body = local_var_pattern.sub("", body)
        expanded_vars: list[tuple[str, str]] = []
        for name, expr in raw_vars:
            for prev_name, prev_expr in expanded_vars:
                expr = re.sub(rf"\b{re.escape(prev_name)}\b", f"({prev_expr})", expr)
            expanded_vars.append((name, expr))
            new_body = re.sub(rf"\b{re.escape(name)}\b", f"({expr})", new_body)

        return f"{header}{new_body}{footer}"

    return pattern.sub(repl_model, content)


class LModFile(SymbolicModel):
    """Class for LARK .mod files"""

    def __init__(self, content: str, filename="<none>.dyno") -> None:
        super().__init__(content, filename)
        self.content = content
        self.parse()
        self.residuals = self.eval_residuals()

    def parse(self: Self) -> None:

        from lark import Lark

        content = substitute_model_local_vars(self.content)
        self.content = content

        trans = ModFileTransformer()
        parser = Lark(
            modfile_grammar,
            propagate_positions=True,
            parser="lalr",
            transformer=trans,
            cache=True,
            # strict=True,
        )
        # Pre-parse check: detect known unsupported features that may either
        # confuse the grammar or survive parsing but fail later.  Raising here
        # gives a clear UnsupportedFeatureError before Lark even tries.
        from dyno.errors import detect_unsupported_features_preparsed

        pre_err = detect_unsupported_features_preparsed(content)
        if pre_err is not None:
            raise pre_err

        try:
            tree = parser.parse(content)
            self.tree = tree
        except UnexpectedInput as e:
            from dyno.errors import detect_unsupported_dynare_feature

            unsupported = detect_unsupported_dynare_feature(e, content)
            if unsupported is not None:
                raise unsupported from e
            raise LARKParserError(e, content) from e

        variables = trans.variables
        variables_exo = trans.variables_exo
        parameters = trans.parameters

        endogenous = [v for v in variables if v not in variables_exo]

        self.symbols = {
            "variables": variables,
            "endogenous": endogenous,
            "exogenous": variables_exo,
            "parameters": parameters,
        }

        self.process_assignments()

    def process_assignments(self, **calib) -> None:

        ## calib ignored so far
        import math

        function_table = {
            "exp": math.exp,
            "log": math.log,
            "sqrt": math.sqrt,
            "abs": math.fabs,
        }

        fe = InterpretModfile(
            symbols=self.symbols, **calib, function_table=function_table
        )
        fe.visit(self.tree)

        self.equations = fe.equations

        # Set processes
        from dyno.language import Normal

        covs = fe.covariances
        exo = tuple(self.symbols["exogenous"])
        n_e = len(exo)
        mat = np.zeros((n_e, n_e))
        for ind, k in covs.items():
            i = exo.index(ind[0])
            j = exo.index(ind[1])
            mat[i, j] = k
            mat[j, i] = k

        processes = {exo: Normal(mat)}

        context = {
            "constants": fe.constants,
            "variables": fe.variables,
            "values": fe.values,
            "processes": processes,
            "steady_states": fe.steady_states
            | {e: 0.0 for e in exo},  # set exogenous steady states to zero
        }

        self.context = context
        self.metadata = {
            "dynare_commands": self._extract_dynare_commands(),
        }

    def _extract_dynare_commands(self: Self) -> list[dict[str, Any]]:
        """Extract Dynare command statements from the parsed modfile tree."""

        def _name_from_node(node: Any) -> str | None:
            if (
                isinstance(node, Tree)
                and node.data == "name"
                and len(node.children) > 0
            ):
                return str(node.children[0])
            return None

        def _coerce_value(node: Any) -> Any:
            if isinstance(node, Token):
                if node.type == "NUMBER":
                    text = str(node)
                    try:
                        if any(ch in text for ch in ".eE"):
                            return float(text)
                        return int(text)
                    except ValueError:
                        return text
                return str(node)
            if isinstance(node, Tree):
                name = _name_from_node(node)
                if name is not None:
                    return name
            return str(node)

        commands: list[dict[str, Any]] = []
        for statement in self.tree.children:
            if not isinstance(statement, Tree) or statement.data != "command_statement":
                continue
            if len(statement.children) == 0:
                continue

            first = statement.children[0]
            if not isinstance(first, Token) or first.type != "COMMAND":
                continue

            command = str(first)
            options: dict[str, Any] = {}

            tail = list(statement.children[1:])
            i = 0
            while i < len(tail):
                child = tail[i]
                if isinstance(child, Tree):
                    if child.data in ("cmd_opt_num", "cmd_opt_name"):
                        k = _name_from_node(child.children[0])
                        v = _coerce_value(child.children[1])
                        if k is not None:
                            options[k] = v
                        i += 1
                        continue
                    elif child.data == "cmd_opt_flag":
                        k = _name_from_node(child.children[0])
                        if k is not None:
                            options[k] = True
                        i += 1
                        continue
                    elif child.data == "name":
                        if i + 1 < len(tail):
                            k = _name_from_node(tail[i])
                            if k is not None:
                                options[k] = _coerce_value(tail[i + 1])
                            i += 2
                            continue
                i += 1

            commands.append({"command": command, "options": options})

        return commands
