"""Self-contained, elegant Lark-based macro processor for Dynare mod files.

Matches the behavior and output of Dynare's preprocessor (`-macroexpand` / `savemacro`).
Implemented compactly and elegantly using Lark grammar and AST evaluation.
"""

from __future__ import annotations

import dataclasses
import functools
import itertools
import math
import os
import re
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

from lark import Lark, Tree, Token
from lark.exceptions import LarkError


class MacroError(Exception):
    """Base exception for macro processing errors."""

    def __init__(
        self, message: str, line: Optional[int] = None, filename: Optional[str] = None
    ):
        self.message = message
        self.line = line
        self.filename = filename
        prefix = ""
        if filename:
            prefix += f"in file {filename}: "
        if line is not None:
            prefix += f"line {line}: "
        super().__init__(f"{prefix}{message}")


class MacroSyntaxError(MacroError):
    """Syntax error during macro parsing."""

    pass


class MacroEvaluationError(MacroError):
    """Runtime / evaluation error during macro execution."""

    pass


# ---------------------------------------------------------------------------
# Values and Types
# ---------------------------------------------------------------------------


class BaseValue:
    """Base class for all macro values."""

    def to_bool(self) -> bool:
        raise MacroEvaluationError(f"Cannot convert {type(self).__name__} to boolean")

    def to_string(self) -> str:
        raise MacroEvaluationError(f"Cannot convert {type(self).__name__} to string")

    def format_output(self) -> str:
        """Format for output insertion into mod file."""
        return self.to_string()


@dataclasses.dataclass(frozen=True)
class RealVal(BaseValue):
    val: Union[int, float]

    def to_bool(self) -> bool:
        return bool(self.val)

    def to_string(self) -> str:
        if isinstance(self.val, int):
            return str(self.val)
        return f"{self.val:.17g}"

    def __repr__(self) -> str:
        return f"RealVal({self.to_string()})"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, RealVal):
            return self.val == other.val
        return False

    def __hash__(self) -> int:
        return hash(self.val)


@dataclasses.dataclass(frozen=True)
class BoolVal(BaseValue):
    val: bool

    def to_bool(self) -> bool:
        return self.val

    def to_string(self) -> str:
        return "true" if self.val else "false"

    def __repr__(self) -> str:
        return f"BoolVal({self.val})"


@dataclasses.dataclass(frozen=True)
class StringVal(BaseValue):
    val: str

    def to_bool(self) -> bool:
        raise MacroEvaluationError("Strings cannot be evaluated as booleans")

    def to_string(self) -> str:
        return self.val

    def format_output(self) -> str:
        return self.val

    def __repr__(self) -> str:
        return f"StringVal({self.val!r})"


@dataclasses.dataclass(frozen=True)
class TupleVal(BaseValue):
    elements: Tuple[BaseValue, ...]

    def to_bool(self) -> bool:
        raise MacroEvaluationError("Tuples cannot be evaluated as booleans")

    def to_string(self) -> str:
        inner = ", ".join(
            f'"{e.to_string()}"' if isinstance(e, StringVal) else e.to_string()
            for e in self.elements
        )
        return f"({inner})"

    def __repr__(self) -> str:
        return f"TupleVal({self.elements!r})"


@dataclasses.dataclass(frozen=True)
class ArrayVal(BaseValue):
    elements: Tuple[BaseValue, ...]

    def to_bool(self) -> bool:
        raise MacroEvaluationError("Arrays cannot be evaluated as booleans")

    def to_string(self) -> str:
        inner = ", ".join(
            f'"{e.to_string()}"' if isinstance(e, StringVal) else e.to_string()
            for e in self.elements
        )
        return f"[{inner}]"

    def __repr__(self) -> str:
        return f"ArrayVal({self.elements!r})"


@dataclasses.dataclass(frozen=True)
class FunctionVal(BaseValue):
    name: str
    args: Tuple[str, ...]
    body_tree: Any
    closure_env: MacroEnvironment

    def to_bool(self) -> bool:
        raise MacroEvaluationError("Functions cannot be evaluated as booleans")

    def to_string(self) -> str:
        return f"<function {self.name}({', '.join(self.args)})>"


def wrap_value(val: Any) -> BaseValue:
    if isinstance(val, BaseValue):
        return val
    if isinstance(val, bool):
        return BoolVal(val)
    if isinstance(val, (int, float)):
        return RealVal(val)
    if isinstance(val, str):
        return StringVal(val)
    if isinstance(val, tuple):
        return TupleVal(tuple(wrap_value(x) for x in val))
    if isinstance(val, list):
        return ArrayVal(tuple(wrap_value(x) for x in val))
    raise MacroEvaluationError(f"Unsupported python value for macro processor: {val!r}")


# ---------------------------------------------------------------------------
# Lark Grammar & Parser
# ---------------------------------------------------------------------------

EXPR_GRAMMAR = r"""
?start: expr

?expr: or_expr

?or_expr: or_expr "||" and_expr -> or_op
        | and_expr

?and_expr: and_expr "&&" not_expr -> and_op
         | not_expr

?not_expr: "!" not_expr -> not_op
         | comp_expr

?comp_expr: comp_expr COMP_OP range_expr -> comp_op
          | range_expr
COMP_OP: "==" | "!=" | "<=" | ">=" | "<" | ">" | "in"

?range_expr: bitor_expr (":" bitor_expr (":" bitor_expr)? ) -> range_op
           | bitor_expr

?bitor_expr: bitor_expr "|" bitand_expr -> bitor_op
           | bitand_expr

?bitand_expr: bitand_expr "&" sum_expr -> bitand_op
            | sum_expr

?sum_expr: sum_expr ADD_OP prod_expr -> sum_op
         | prod_expr
ADD_OP: "+" | "-"

?prod_expr: prod_expr MUL_OP power_expr -> prod_op
          | power_expr
MUL_OP: "*" | "/"

?power_expr: unary_expr "^" power_expr -> pow_op
           | unary_expr

?unary_expr: "+" unary_expr -> pos_op
           | "-" unary_expr -> neg_op
           | postfix_expr

?postfix_expr: atom
             | postfix_expr "[" expr "]" -> index_op
             | postfix_expr "(" [expr_list] ")" -> call_op

expr_list: expr ("," expr)*

?atom: NUMBER -> number
     | STRING -> string
     | "true" -> true_val
     | "false" -> false_val
     | "defined" "(" NAME ")" -> defined_op
     | NAME -> var
     | "(" expr ")"
     | "(" expr "," [expr_list] ")" -> tuple_lit
     | "(" ")" -> empty_tuple
     | "[" comprehension "]"
     | "[" [expr_list] "]" -> array_lit

?comprehension: expr "for" loop_target "in" expr ["when" expr] -> comp_expr1
              | loop_target "in" expr "when" expr -> comp_expr2

loop_target: NAME -> single_var
           | "(" NAME ("," NAME)+ ")" -> tuple_vars

NAME: /[a-zA-Z_][a-zA-Z0-9_]*/
NUMBER: /(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?|\.\d+(?:[eE][+-]?\d+)?/
STRING: /"([^"\\]|\\.)*"/
%import common.WS
%ignore WS
"""


@functools.cache
def _lark_parser() -> Lark:
    return Lark(EXPR_GRAMMAR, parser="earley")


def parse_expression(text: str) -> Tree:
    try:
        return _lark_parser().parse(text)
    except LarkError as e:
        raise MacroSyntaxError(f"Expression syntax error: {e}") from e


# ---------------------------------------------------------------------------
# Builtin Functions
# ---------------------------------------------------------------------------


def _builtin_unary_real(
    name: str, fn: Callable[[float], float]
) -> Callable[[List[BaseValue]], BaseValue]:
    def handler(args: List[BaseValue]) -> BaseValue:
        if len(args) != 1:
            raise MacroEvaluationError(
                f"{name}() takes exactly 1 argument ({len(args)} given)"
            )
        v = args[0]
        if not isinstance(v, RealVal):
            raise MacroEvaluationError(
                f"{name}() argument must be a Real, got {type(v).__name__}"
            )
        try:
            return RealVal(fn(float(v.val)))
        except (ValueError, OverflowError, ZeroDivisionError) as e:
            raise MacroEvaluationError(f"{name}() evaluation error: {e}")

    return handler


def _builtin_round(args: List[BaseValue]) -> BaseValue:
    if len(args) != 1 or not isinstance(args[0], RealVal):
        raise MacroEvaluationError("round() takes exactly 1 Real argument")
    x = float(args[0].val)
    return RealVal(int(math.floor(x + 0.5) if x >= 0 else math.ceil(x - 0.5)))


def _builtin_sign(args: List[BaseValue]) -> BaseValue:
    if len(args) != 1 or not isinstance(args[0], RealVal):
        raise MacroEvaluationError("sign() takes exactly 1 Real argument")
    x = args[0].val
    return RealVal(1 if x > 0 else (-1 if x < 0 else 0))


def _builtin_normpdf(args: List[BaseValue]) -> BaseValue:
    if len(args) == 1:
        x_val, mu, sigma = args[0], 0.0, 1.0
    elif len(args) == 3:
        x_val, mu_val, sig_val = args
        if not isinstance(mu_val, RealVal) or not isinstance(sig_val, RealVal):
            raise MacroEvaluationError("normpdf() mu and sigma must be Real")
        mu, sigma = float(mu_val.val), float(sig_val.val)
    else:
        raise MacroEvaluationError(
            f"normpdf() takes 1 or 3 arguments ({len(args)} given)"
        )

    if not isinstance(x_val, RealVal) or sigma <= 0:
        raise MacroEvaluationError("normpdf() invalid arguments")
    z = (float(x_val.val) - mu) / sigma
    return RealVal(math.exp(-0.5 * z * z) / (sigma * math.sqrt(2.0 * math.pi)))


def _builtin_normcdf(args: List[BaseValue]) -> BaseValue:
    if len(args) == 1:
        x_val, mu, sigma = args[0], 0.0, 1.0
    elif len(args) == 3:
        x_val, mu_val, sig_val = args
        if not isinstance(mu_val, RealVal) or not isinstance(sig_val, RealVal):
            raise MacroEvaluationError("normcdf() mu and sigma must be Real")
        mu, sigma = float(mu_val.val), float(sig_val.val)
    else:
        raise MacroEvaluationError(
            f"normcdf() takes 1 or 3 arguments ({len(args)} given)"
        )

    if not isinstance(x_val, RealVal) or sigma <= 0:
        raise MacroEvaluationError("normcdf() invalid arguments")
    z = (float(x_val.val) - mu) / sigma
    return RealVal(0.5 * (1.0 + math.erf(z / math.sqrt(2.0))))


def _builtin_extremum(is_max: bool) -> Callable[[List[BaseValue]], BaseValue]:
    name = "max" if is_max else "min"

    def handler(args: List[BaseValue]) -> BaseValue:
        if not args:
            raise MacroEvaluationError(f"{name}() takes at least 1 argument")
        items = (
            args[0].elements
            if len(args) == 1 and isinstance(args[0], ArrayVal)
            else tuple(args)
        )
        if not items:
            raise MacroEvaluationError(f"{name}() of empty array")
        real_vals: List[Union[int, float]] = []
        for x in items:
            if not isinstance(x, RealVal):
                raise MacroEvaluationError(f"{name}() elements must be Real")
            real_vals.append(x.val)
        fn = max if is_max else min
        return RealVal(fn(real_vals))

    return handler


def _builtin_length(args: List[BaseValue]) -> BaseValue:
    if len(args) != 1:
        raise MacroEvaluationError(
            f"length() takes exactly 1 argument ({len(args)} given)"
        )
    v = args[0]
    if isinstance(v, (ArrayVal, TupleVal)):
        return RealVal(len(v.elements))
    if isinstance(v, StringVal):
        return RealVal(len(v.val))
    raise MacroEvaluationError(f"length() not supported for {type(v).__name__}")


def _builtin_isempty(args: List[BaseValue]) -> BaseValue:
    if len(args) != 1:
        raise MacroEvaluationError(
            f"isEmpty() takes exactly 1 argument ({len(args)} given)"
        )
    v = args[0]
    if isinstance(v, (ArrayVal, TupleVal)):
        return BoolVal(len(v.elements) == 0)
    if isinstance(v, StringVal):
        return BoolVal(len(v.val) == 0)
    raise MacroEvaluationError(f"isEmpty() not supported for {type(v).__name__}")


def _builtin_sum(args: List[BaseValue]) -> BaseValue:
    if len(args) != 1 or not isinstance(args[0], ArrayVal):
        raise MacroEvaluationError("sum() requires an Array")
    elems = args[0].elements
    if not elems:
        return RealVal(0)
    first = elems[0]
    if isinstance(first, RealVal):
        num_total: Union[int, float] = 0
        for e in elems:
            if not isinstance(e, RealVal):
                raise MacroEvaluationError("All elements of sum() array must be Real")
            num_total += e.val
        return RealVal(num_total)
    elif isinstance(first, StringVal):
        s_total = ""
        for e in elems:
            if not isinstance(e, StringVal):
                raise MacroEvaluationError("All elements of sum() array must be String")
            s_total += e.val
        return StringVal(s_total)
    elif isinstance(first, ArrayVal):
        a_total: List[BaseValue] = []
        for e in elems:
            if not isinstance(e, ArrayVal):
                raise MacroEvaluationError("All elements of sum() array must be Array")
            a_total.extend(e.elements)
        return ArrayVal(tuple(a_total))
    raise MacroEvaluationError(
        f"sum() not supported for array of {type(first).__name__}"
    )


def _builtin_diff(args: List[BaseValue]) -> BaseValue:
    if len(args) != 1 or not isinstance(args[0], ArrayVal) or not args[0].elements:
        raise MacroEvaluationError("diff() requires a non-empty Array")
    elems = args[0].elements
    first = elems[0]
    if not isinstance(first, RealVal):
        raise MacroEvaluationError("diff() elements must be Real")
    res = first.val
    for e in elems[1:]:
        if not isinstance(e, RealVal):
            raise MacroEvaluationError("diff() elements must be Real")
        res -= e.val
    return RealVal(res)


def _builtin_prod(args: List[BaseValue]) -> BaseValue:
    if len(args) != 1 or not isinstance(args[0], ArrayVal):
        raise MacroEvaluationError("prod() requires an Array")
    elems = args[0].elements
    if not elems:
        return RealVal(1)
    cur: Union[int, float] = 1
    for e in elems:
        if not isinstance(e, RealVal):
            raise MacroEvaluationError("prod() elements must be Real")
        cur *= e.val
    return RealVal(cur)


BUILTIN_FUNCS: Dict[str, Callable[[List[BaseValue]], BaseValue]] = {
    "exp": _builtin_unary_real("exp", math.exp),
    "ln": _builtin_unary_real("ln", math.log),
    "log": _builtin_unary_real("log", math.log),
    "log2": _builtin_unary_real("log2", math.log2),
    "log10": _builtin_unary_real("log10", math.log10),
    "sqrt": _builtin_unary_real("sqrt", math.sqrt),
    "cbrt": _builtin_unary_real(
        "cbrt", math.cbrt if hasattr(math, "cbrt") else lambda x: x ** (1.0 / 3.0)
    ),
    "sign": _builtin_sign,
    "floor": _builtin_unary_real("floor", math.floor),
    "ceil": _builtin_unary_real("ceil", math.ceil),
    "trunc": _builtin_unary_real("trunc", math.trunc),
    "round": _builtin_round,
    "erf": _builtin_unary_real("erf", math.erf),
    "erfc": _builtin_unary_real("erfc", math.erfc),
    "gamma": _builtin_unary_real("gamma", math.gamma),
    "lgamma": _builtin_unary_real("lgamma", math.lgamma),
    "normpdf": _builtin_normpdf,
    "normcdf": _builtin_normcdf,
    "max": _builtin_extremum(is_max=True),
    "min": _builtin_extremum(is_max=False),
    "length": _builtin_length,
    "isEmpty": _builtin_isempty,
    "sum": _builtin_sum,
    "diff": _builtin_diff,
    "prod": _builtin_prod,
}


# ---------------------------------------------------------------------------
# AST Evaluator
# ---------------------------------------------------------------------------


def eval_tree(node: Union[Tree, Token], env: MacroEnvironment) -> BaseValue:
    if isinstance(node, Token):
        raise MacroEvaluationError(f"Unexpected raw token in evaluation: {node}")

    rule = node.data
    ch = node.children

    if rule == "number":
        s = ch[0].value  # type: ignore[union-attr]
        val: Union[int, float] = (
            int(s) if ("." not in s and "e" not in s and "E" not in s) else float(s)
        )
        return RealVal(val)

    if rule == "string":
        raw = ch[0].value[1:-1]  # type: ignore[union-attr]
        return StringVal(raw.replace(r"\"", '"').replace(r"\\", "\\"))

    if rule == "true_val":
        return BoolVal(True)

    if rule == "false_val":
        return BoolVal(False)

    if rule == "var":
        return env.get(ch[0].value)  # type: ignore[union-attr]

    if rule == "defined_op":
        return BoolVal(env.is_defined(ch[0].value))  # type: ignore[union-attr]

    if rule == "tuple_lit":
        first = eval_tree(ch[0], env)
        rest = [eval_tree(x, env) for x in ch[1].children] if len(ch) > 1 and ch[1] is not None else []  # type: ignore[union-attr]
        return TupleVal(tuple([first] + rest))

    if rule == "empty_tuple":
        return TupleVal(())

    if rule == "array_lit":
        if not ch or ch[0] is None:
            return ArrayVal(())
        return ArrayVal(tuple(eval_tree(x, env) for x in ch[0].children))  # type: ignore[union-attr]

    if rule == "range_op":
        start_v = eval_tree(ch[0], env)
        if len(ch) == 2:
            stop_v = eval_tree(ch[1], env)
            step = 1
        else:
            step_v = eval_tree(ch[1], env)
            stop_v = eval_tree(ch[2], env)
            if (
                not isinstance(step_v, RealVal)
                or not isinstance(step_v.val, int)
                or step_v.val == 0
            ):
                raise MacroEvaluationError("Range step must be a non-zero integer")
            step = step_v.val
        if not (isinstance(start_v, RealVal) and isinstance(start_v.val, int)) or not (
            isinstance(stop_v, RealVal) and isinstance(stop_v.val, int)
        ):
            raise MacroEvaluationError("Range start and stop must be integers")
        r_range = range(start_v.val, stop_v.val + (1 if step > 0 else -1), step)
        return ArrayVal(tuple(RealVal(x) for x in r_range))

    if rule == "comp_expr1":
        expr_ast, target_ast, iter_ast = ch[0], ch[1], ch[2]
        when_ast = ch[3] if len(ch) > 3 else None
        arr_val = eval_tree(iter_ast, env)
        if not isinstance(arr_val, ArrayVal):
            raise MacroEvaluationError("Comprehension iterable must be an Array")
        res: List[BaseValue] = []
        is_single = target_ast.data == "single_var"  # type: ignore[union-attr]
        var_names = [target_ast.children[0].value] if is_single else [t.value for t in target_ast.children]  # type: ignore[union-attr]
        for item in arr_val.elements:
            subenv = MacroEnvironment(parent=env)
            if is_single:
                subenv.set(var_names[0], item)
            else:
                if not isinstance(item, TupleVal) or len(item.elements) != len(
                    var_names
                ):
                    raise MacroEvaluationError("Cannot unpack tuple in comprehension")
                for k, v in zip(var_names, item.elements):
                    subenv.set(k, v)
            if when_ast is not None:
                cond = eval_tree(when_ast, subenv)
                if not cond.to_bool():
                    continue
            res.append(eval_tree(expr_ast, subenv))
        return ArrayVal(tuple(res))

    if rule == "comp_expr2":
        target_ast, iter_ast, when_ast = ch[0], ch[1], ch[2]
        arr_val = eval_tree(iter_ast, env)
        if not isinstance(arr_val, ArrayVal):
            raise MacroEvaluationError("Comprehension iterable must be an Array")
        res = []
        is_single = target_ast.data == "single_var"  # type: ignore[union-attr]
        var_names = [target_ast.children[0].value] if is_single else [t.value for t in target_ast.children]  # type: ignore[union-attr]
        for item in arr_val.elements:
            subenv = MacroEnvironment(parent=env)
            if is_single:
                subenv.set(var_names[0], item)
            else:
                if not isinstance(item, TupleVal) or len(item.elements) != len(
                    var_names
                ):
                    raise MacroEvaluationError("Cannot unpack tuple in comprehension")
                for k, v in zip(var_names, item.elements):
                    subenv.set(k, v)
            if eval_tree(when_ast, subenv).to_bool():
                res.append(item)
        return ArrayVal(tuple(res))

    if rule == "call_op":
        fn_node = ch[0]
        args: List[BaseValue] = (
            [eval_tree(x, env) for x in ch[1].children]  # type: ignore[union-attr]
            if len(ch) > 1 and ch[1] is not None
            else []
        )
        if isinstance(fn_node, Tree) and fn_node.data == "var" and fn_node.children[0].value in BUILTIN_FUNCS:  # type: ignore[union-attr]
            return BUILTIN_FUNCS[fn_node.children[0].value](args)  # type: ignore[union-attr]
        fval = eval_tree(fn_node, env)
        if isinstance(fval, FunctionVal):
            if len(args) != len(fval.args):
                raise MacroEvaluationError(
                    f"Function {fval.name} expects {len(fval.args)} args, got {len(args)}"
                )
            call_env = MacroEnvironment(parent=fval.closure_env)
            for param, val_arg in zip(fval.args, args):
                call_env.set(param, val_arg)
            return eval_tree(fval.body_tree, call_env)
        raise MacroEvaluationError(f"{type(fval).__name__} is not callable")

    if rule == "index_op":
        target = eval_tree(ch[0], env)
        idx_v = eval_tree(ch[1], env)
        if not isinstance(idx_v, RealVal) or not isinstance(idx_v.val, int):
            raise MacroEvaluationError("Index must be an integer")
        idx = idx_v.val
        if isinstance(target, (ArrayVal, TupleVal)):
            if idx < 1 or idx > len(target.elements):
                raise MacroEvaluationError(f"Index {idx} out of bounds")
            return target.elements[idx - 1]
        if isinstance(target, StringVal):
            if idx < 1 or idx > len(target.val):
                raise MacroEvaluationError(f"Index {idx} out of bounds")
            return StringVal(target.val[idx - 1])
        raise MacroEvaluationError(f"Cannot index into {type(target).__name__}")

    if rule == "pos_op":
        v = eval_tree(ch[0], env)
        if not isinstance(v, RealVal):
            raise MacroEvaluationError("Unary + requires Real")
        return v

    if rule == "neg_op":
        v = eval_tree(ch[0], env)
        if not isinstance(v, RealVal):
            raise MacroEvaluationError("Unary - requires Real")
        return RealVal(-v.val)

    if rule == "not_op":
        v = eval_tree(ch[0], env)
        return BoolVal(not v.to_bool())

    if rule == "pow_op":
        base = eval_tree(ch[0], env)
        exp = eval_tree(ch[1], env)
        if isinstance(base, RealVal) and isinstance(exp, RealVal):
            p_res = base.val**exp.val
            if isinstance(p_res, complex):
                raise MacroEvaluationError("Power resulted in complex number")
            return RealVal(
                int(p_res)
                if (
                    isinstance(base.val, int)
                    and isinstance(exp.val, int)
                    and exp.val >= 0
                )
                else float(p_res)
            )
        if (
            isinstance(base, ArrayVal)
            and isinstance(exp, RealVal)
            and isinstance(exp.val, int)
            and exp.val >= 1
        ):
            curr = base
            for _ in range(exp.val - 1):
                prod_elems: List[BaseValue] = []
                for a in curr.elements:
                    for b in base.elements:
                        prod_elems.append(
                            TupleVal(a.elements + (b,))
                            if isinstance(a, TupleVal)
                            else TupleVal((a, b))
                        )
                curr = ArrayVal(tuple(prod_elems))
            return curr
        raise MacroEvaluationError(
            f"Unsupported operand types for ^: {type(base).__name__} and {type(exp).__name__}"
        )

    if rule == "prod_op":
        lval = eval_tree(ch[0], env)
        op = ch[1].value  # type: ignore[union-attr]
        rval = eval_tree(ch[2], env)
        if op == "*":
            if isinstance(lval, RealVal) and isinstance(rval, RealVal):
                return RealVal(lval.val * rval.val)
            if isinstance(lval, ArrayVal) and isinstance(rval, ArrayVal):
                res_cart: List[BaseValue] = []
                for a in lval.elements:
                    for b in rval.elements:
                        res_cart.append(
                            TupleVal(a.elements + (b,))
                            if isinstance(a, TupleVal)
                            else TupleVal((a, b))
                        )
                return ArrayVal(tuple(res_cart))
        elif op == "/":
            if isinstance(lval, RealVal) and isinstance(rval, RealVal):
                if rval.val == 0:
                    raise MacroEvaluationError("Division by zero")
                if (
                    isinstance(lval.val, int)
                    and isinstance(rval.val, int)
                    and lval.val % rval.val == 0
                ):
                    return RealVal(lval.val // rval.val)
                return RealVal(lval.val / rval.val)
        raise MacroEvaluationError(
            f"Operator {op} not supported between {type(lval).__name__} and {type(rval).__name__}"
        )

    if rule == "sum_op":
        lval = eval_tree(ch[0], env)
        op = ch[1].value  # type: ignore[union-attr]
        rval = eval_tree(ch[2], env)
        if op == "+":
            if isinstance(lval, RealVal) and isinstance(rval, RealVal):
                return RealVal(lval.val + rval.val)
            if isinstance(lval, StringVal) and isinstance(rval, StringVal):
                return StringVal(lval.val + rval.val)
            if isinstance(lval, ArrayVal) and isinstance(rval, ArrayVal):
                return ArrayVal(lval.elements + rval.elements)
            if isinstance(lval, TupleVal) and isinstance(rval, TupleVal):
                return TupleVal(lval.elements + rval.elements)
        elif op == "-":
            if isinstance(lval, RealVal) and isinstance(rval, RealVal):
                return RealVal(lval.val - rval.val)
            if isinstance(lval, ArrayVal) and isinstance(rval, ArrayVal):
                r_set = set(rval.elements)
                return ArrayVal(tuple(x for x in lval.elements if x not in r_set))
        raise MacroEvaluationError(
            f"Operator {op} not supported between {type(lval).__name__} and {type(rval).__name__}"
        )

    if rule == "bitand_op":
        lval = eval_tree(ch[0], env)
        rval = eval_tree(ch[1], env)
        if isinstance(lval, ArrayVal) and isinstance(rval, ArrayVal):
            l_set = set(lval.elements)
            return ArrayVal(tuple(item for item in rval.elements if item in l_set))
        raise MacroEvaluationError("Arguments of intersection (&) must be Array")

    if rule == "bitor_op":
        lval = eval_tree(ch[0], env)
        rval = eval_tree(ch[1], env)
        if isinstance(lval, ArrayVal) and isinstance(rval, ArrayVal):
            new_elems = list(lval.elements)
            for item in rval.elements:
                if item not in new_elems:
                    new_elems.append(item)
            return ArrayVal(tuple(new_elems))
        raise MacroEvaluationError("Arguments of union (|) must be Array")

    if rule == "comp_op":
        lval = eval_tree(ch[0], env)
        op = ch[1].value  # type: ignore[union-attr]
        rval = eval_tree(ch[2], env)
        if op == "==":
            return BoolVal(lval == rval)
        if op == "!=":
            return BoolVal(lval != rval)
        if op in ("<", "<=", ">", ">="):
            if (
                isinstance(lval, (RealVal, StringVal))
                and isinstance(rval, (RealVal, StringVal))
                and type(lval) is type(rval)
            ):
                c = (
                    lval.val < rval.val  # type: ignore[operator]
                    if op == "<"
                    else (
                        lval.val <= rval.val  # type: ignore[operator]
                        if op == "<="
                        else (lval.val > rval.val if op == ">" else lval.val >= rval.val)  # type: ignore[operator]
                    )
                )
                return BoolVal(c)
            raise MacroEvaluationError(
                f"Comparison {op} not supported between {type(lval).__name__} and {type(rval).__name__}"
            )
        if op == "in":
            if isinstance(rval, (ArrayVal, TupleVal)):
                return BoolVal(any(lval == elem for elem in rval.elements))
            raise MacroEvaluationError("'in' right operand must be Array or Tuple")

    if rule == "and_op":
        lval = eval_tree(ch[0], env)
        if not lval.to_bool():
            return BoolVal(False)
        rval = eval_tree(ch[1], env)
        return BoolVal(rval.to_bool())

    if rule == "or_op":
        lval = eval_tree(ch[0], env)
        if lval.to_bool():
            return BoolVal(True)
        rval = eval_tree(ch[1], env)
        return BoolVal(rval.to_bool())

    raise MacroEvaluationError(f"Unknown AST rule: {rule}")


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


class MacroEnvironment:
    def __init__(
        self,
        parent: Optional[MacroEnvironment] = None,
        initial_vars: Optional[Dict[str, Any]] = None,
    ):
        self.parent = parent
        self.vars: Dict[str, BaseValue] = {}
        if initial_vars:
            for k, v in initial_vars.items():
                self.vars[k] = wrap_value(v)

    def get(self, name: str) -> BaseValue:
        if name in self.vars:
            return self.vars[name]
        if self.parent is not None:
            return self.parent.get(name)
        raise MacroEvaluationError(f"Undefined macro variable: {name}")

    def is_defined(self, name: str) -> bool:
        if name in self.vars:
            return True
        if self.parent is not None:
            return self.parent.is_defined(name)
        return False

    def set(self, name: str, val: BaseValue) -> None:
        self.vars[name] = val

    def all_vars(self) -> Dict[str, BaseValue]:
        res: Dict[str, BaseValue] = {}
        if self.parent is not None:
            res.update(self.parent.all_vars())
        res.update(self.vars)
        return res


# ---------------------------------------------------------------------------
# Macro Processor Engine
# ---------------------------------------------------------------------------


DIRECTIVE_RE = re.compile(r"(?:^\s*@#|@\{)", re.MULTILINE)
CONTINUATION_RE = re.compile(r"\\\\\s*$", re.MULTILINE)


def has_macro_directives(text: str) -> bool:
    """Return True if text contains macro directives, inline substitutions, or continuations."""
    return bool(DIRECTIVE_RE.search(text) or CONTINUATION_RE.search(text))


class MacroProcessor:
    """Dynare-compatible macro processor using Lark."""

    def __init__(
        self,
        main_dir: str = ".",
        include_paths: Optional[Sequence[str]] = None,
        defines: Optional[Dict[str, Any]] = None,
    ):
        self.main_dir = os.path.abspath(main_dir)
        self.include_paths = [self.main_dir]
        if include_paths:
            for p in include_paths:
                self.include_paths.append(os.path.abspath(p))
        self.env = MacroEnvironment(initial_vars=defines)
        self.included_files: List[str] = []

    def expand_file(self, filename: str, only_if_needed: bool = False) -> str:
        abs_path = os.path.abspath(filename)
        self.main_dir = os.path.dirname(abs_path)
        if self.main_dir not in self.include_paths:
            self.include_paths.insert(0, self.main_dir)
        with open(abs_path, "rt", encoding="utf-8") as f:
            content = f.read()
        return self.expand_string(
            content, filename=abs_path, only_if_needed=only_if_needed
        )

    def expand_string(
        self, text: str, filename: str = "<string>", only_if_needed: bool = False
    ) -> str:
        if only_if_needed and not has_macro_directives(text):
            return text
        lines = self._join_continuations(text)
        raw_output_lines = self._process_lines(lines, self.env, filename)
        raw_output = "\n".join(raw_output_lines) + "\n"
        return self.postprocess_output(raw_output)

    @classmethod
    def postprocess_output(cls, text: str) -> str:
        """Applies Dynare savemacro output cleanup."""
        text = re.sub(r"(?m)^@#line.*$", "", text)
        text = re.sub(r"^(\r?\n)+", "", text)
        text = re.sub(r"\n{2,}", "\n", text)
        text = re.sub(r"(\r\n){2,}", "\r\n", text)
        return text

    def _join_continuations(self, text: str) -> List[Tuple[int, str]]:
        raw_lines = text.splitlines()
        result: List[Tuple[int, str]] = []
        i = 0
        n = len(raw_lines)
        while i < n:
            start_line = i + 1
            line = raw_lines[i]
            while line.endswith(r"\\") and i + 1 < n:
                line = line[:-2] + raw_lines[i + 1]
                i += 1
            result.append((start_line, line))
            i += 1
        return result

    def _process_lines(
        self,
        lines: List[Tuple[int, str]],
        env: MacroEnvironment,
        filename: str,
    ) -> List[str]:
        output_lines: List[str] = []
        i = 0
        n = len(lines)

        while i < n:
            line_no, line = lines[i]
            stripped = line.strip()

            if stripped.startswith("@#"):
                dir_content = re.sub(r"//.*$", "", stripped[2:]).strip()

                if dir_content.startswith("include"):
                    self._handle_include(
                        dir_content[7:].strip(), line_no, filename, env, output_lines
                    )
                    i += 1
                elif dir_content.startswith("define"):
                    self._handle_define(dir_content[6:].strip(), line_no, filename, env)
                    i += 1
                elif dir_content.startswith("echo"):
                    self._handle_echo(dir_content[4:].strip(), line_no, filename, env)
                    i += 1
                elif dir_content.startswith("echomacrovars"):
                    self._handle_echomacrovars(env)
                    i += 1
                elif dir_content.startswith("error"):
                    self._handle_error(dir_content[5:].strip(), line_no, filename, env)
                    i += 1
                elif dir_content.startswith("for"):
                    block_lines, next_i = self._extract_for_block(lines, i)
                    for_output = self._process_for_block(block_lines, env, filename)
                    output_lines.extend(for_output)
                    i = next_i
                elif (
                    dir_content.startswith("if")
                    or dir_content.startswith("ifdef")
                    or dir_content.startswith("ifndef")
                ):
                    branch_lines, next_i = self._extract_if_branches(
                        lines, i, env, filename
                    )
                    if branch_lines:
                        branch_output = self._process_lines(branch_lines, env, filename)
                        output_lines.extend(branch_output)
                    i = next_i
                else:
                    raise MacroSyntaxError(
                        f"Unknown or misplaced directive: @#{dir_content}",
                        line=line_no,
                        filename=filename,
                    )
            else:
                eval_line = self._eval_inline(line, env, line_no, filename)
                output_lines.append(eval_line)
                i += 1

        return output_lines

    def _eval_inline(
        self, line: str, env: MacroEnvironment, line_no: int, filename: str
    ) -> str:
        pos = 0
        length = len(line)
        result = []
        while pos < length:
            idx = line.find("@{", pos)
            if idx == -1:
                result.append(line[pos:])
                break
            result.append(line[pos:idx])
            p = idx + 2
            brace_count = 1
            in_str = False
            expr_chars = []
            while p < length:
                ch = line[p]
                if ch == '"' and (p == 0 or line[p - 1] != "\\"):
                    in_str = not in_str
                    expr_chars.append(ch)
                elif not in_str:
                    if ch == "{":
                        brace_count += 1
                        expr_chars.append(ch)
                    elif ch == "}":
                        brace_count -= 1
                        if brace_count == 0:
                            p += 1
                            break
                        expr_chars.append(ch)
                    else:
                        expr_chars.append(ch)
                else:
                    expr_chars.append(ch)
                p += 1

            if brace_count != 0:
                raise MacroSyntaxError(
                    "Unclosed '@{' in line", line=line_no, filename=filename
                )

            expr_str = "".join(expr_chars).strip()
            try:
                tree = parse_expression(expr_str)
                val = eval_tree(tree, env)
                result.append(val.format_output())
            except Exception as e:
                raise MacroEvaluationError(
                    f"Error evaluating inline expression @{{{expr_str}}}: {e}",
                    line=line_no,
                    filename=filename,
                ) from e

            pos = p

        return "".join(result)

    def _handle_include(
        self,
        rest: str,
        line_no: int,
        filename: str,
        env: MacroEnvironment,
        output_lines: List[str],
    ) -> None:
        rest = rest.strip()
        if not (rest.startswith('"') and rest.endswith('"') and len(rest) >= 2):
            raise MacroSyntaxError(
                f"@#include expects quoted path, got: {rest}",
                line=line_no,
                filename=filename,
            )
        rel_path = rest[1:-1]
        found_path = None
        for inc_dir in self.include_paths:
            candidate = os.path.join(inc_dir, rel_path)
            if os.path.isfile(candidate):
                found_path = candidate
                break

        if not found_path:
            raise MacroEvaluationError(
                f"Included file not found: {rel_path} in {self.include_paths}",
                line=line_no,
                filename=filename,
            )

        with open(found_path, "rt", encoding="utf-8") as f:
            inc_content = f.read()

        inc_lines = self._join_continuations(inc_content)
        inc_output = self._process_lines(inc_lines, env, found_path)
        output_lines.extend(inc_output)

    def _handle_define(
        self, rest: str, line_no: int, filename: str, env: MacroEnvironment
    ) -> None:
        rest = rest.strip()
        match = re.match(
            r"^([A-Za-z_][A-Za-z0-9_]*)(?:\s*\(([^)]*)\))?(?:\s*=(.*))?$",
            rest,
            re.DOTALL,
        )
        if not match:
            raise MacroSyntaxError(
                f"Invalid @#define syntax: {rest}", line=line_no, filename=filename
            )

        name = match.group(1)
        params = match.group(2)
        body = match.group(3)

        if params is not None:
            param_names = tuple(p.strip() for p in params.split(",") if p.strip())
            if body is None:
                raise MacroSyntaxError(
                    f"Function definition @#define {name}(...) must have '=' body",
                    line=line_no,
                    filename=filename,
                )
            tree = parse_expression(body.strip())
            env.set(
                name,
                FunctionVal(
                    name=name, args=param_names, body_tree=tree, closure_env=env
                ),
            )
        else:
            if body is None:
                env.set(name, RealVal(1))
            else:
                tree = parse_expression(body.strip())
                val = eval_tree(tree, env)
                env.set(name, val)

    def _handle_echo(
        self, rest: str, line_no: int, filename: str, env: MacroEnvironment
    ) -> None:
        msg = self._eval_inline(rest, env, line_no, filename)
        print(msg)

    def _handle_echomacrovars(self, env: MacroEnvironment) -> None:
        all_v = env.all_vars()
        print("Macro variables:")
        for k in sorted(all_v.keys()):
            print(f"  {k} = {all_v[k].to_string()}")

    def _handle_error(
        self, rest: str, line_no: int, filename: str, env: MacroEnvironment
    ) -> None:
        msg = self._eval_inline(rest, env, line_no, filename)
        raise MacroEvaluationError(
            f"@#error directive triggered: {msg}", line=line_no, filename=filename
        )

    def _extract_for_block(
        self, lines: List[Tuple[int, str]], start_idx: int
    ) -> Tuple[List[Tuple[int, str]], int]:
        depth = 0
        i = start_idx
        n = len(lines)
        block: List[Tuple[int, str]] = []

        while i < n:
            line_no, line = lines[i]
            stripped = line.strip()
            if stripped.startswith("@#"):
                dir_content = re.sub(r"//.*$", "", stripped[2:]).strip()
                if dir_content.startswith("for"):
                    depth += 1
                elif dir_content.startswith("endfor"):
                    depth -= 1
                    if depth == 0:
                        block.append((line_no, line))
                        return block, i + 1
            block.append((line_no, line))
            i += 1

        start_line = lines[start_idx][0]
        raise MacroSyntaxError(
            "Unmatched @#for directive (missing @#endfor)", line=start_line
        )

    def _process_for_block(
        self, block: List[Tuple[int, str]], env: MacroEnvironment, filename: str
    ) -> List[str]:
        first_line_no, first_line = block[0]
        first_dir = re.sub(r"//.*$", "", first_line.strip()[2:]).strip()
        match = re.match(r"^for\s+(.*?)\s+in\s+(.*)$", first_dir, re.DOTALL)
        if not match:
            raise MacroSyntaxError(
                f"Invalid @#for syntax: {first_dir}",
                line=first_line_no,
                filename=filename,
            )

        loop_vars_str = match.group(1).strip()
        iterable_str = match.group(2).strip()

        if loop_vars_str.startswith("(") and loop_vars_str.endswith(")"):
            loop_vars = [v.strip() for v in loop_vars_str[1:-1].split(",") if v.strip()]
        else:
            loop_vars = [loop_vars_str]

        iter_tree = parse_expression(iterable_str)
        iter_val = eval_tree(iter_tree, env)
        if not isinstance(iter_val, ArrayVal):
            raise MacroEvaluationError(
                f"@#for loop target must be an Array, got {type(iter_val).__name__}",
                line=first_line_no,
                filename=filename,
            )

        inner_lines = block[1:-1]
        output: List[str] = []

        for item in iter_val.elements:
            loop_env = MacroEnvironment(parent=env)
            if len(loop_vars) == 1:
                loop_env.set(loop_vars[0], item)
            else:
                if not isinstance(item, TupleVal) or len(item.elements) != len(
                    loop_vars
                ):
                    raise MacroEvaluationError(
                        f"Cannot unpack {type(item).__name__} into {len(loop_vars)} variables",
                        line=first_line_no,
                        filename=filename,
                    )
                for vname, val in zip(loop_vars, item.elements):
                    loop_env.set(vname, val)

            block_out = self._process_lines(inner_lines, loop_env, filename)
            output.extend(block_out)

        return output

    def _extract_if_branches(
        self,
        lines: List[Tuple[int, str]],
        start_idx: int,
        env: MacroEnvironment,
        filename: str,
    ) -> Tuple[List[Tuple[int, str]], int]:
        depth = 0
        i = start_idx
        n = len(lines)

        branches: List[Tuple[bool, List[Tuple[int, str]]]] = []
        current_branch_lines: List[Tuple[int, str]] = []

        first_line_no, first_line = lines[start_idx]
        first_dir = re.sub(r"//.*$", "", first_line.strip()[2:]).strip()

        if first_dir.startswith("if "):
            cond_tree = parse_expression(first_dir[3:].strip())
            cval = eval_tree(cond_tree, env)
            current_cond_met = cval.to_bool()
        elif first_dir.startswith("ifdef"):
            var_name = first_dir[5:].strip()
            current_cond_met = env.is_defined(var_name)
        elif first_dir.startswith("ifndef"):
            var_name = first_dir[6:].strip()
            current_cond_met = not env.is_defined(var_name)
        else:
            raise MacroSyntaxError(
                f"Expected @#if, @#ifdef or @#ifndef, got {first_dir}",
                line=first_line_no,
                filename=filename,
            )

        i += 1
        depth = 1

        while i < n:
            line_no, line = lines[i]
            stripped = line.strip()

            if stripped.startswith("@#"):
                dir_content = re.sub(r"//.*$", "", stripped[2:]).strip()
                if (
                    dir_content.startswith("if")
                    or dir_content.startswith("ifdef")
                    or dir_content.startswith("ifndef")
                ):
                    depth += 1
                    current_branch_lines.append((line_no, line))
                elif dir_content.startswith("endif"):
                    depth -= 1
                    if depth == 0:
                        branches.append((current_cond_met, current_branch_lines))
                        i += 1
                        break
                    else:
                        current_branch_lines.append((line_no, line))
                elif depth == 1 and (
                    dir_content.startswith("elseif") or dir_content.startswith("elif")
                ):
                    branches.append((current_cond_met, current_branch_lines))
                    current_branch_lines = []
                    expr_str = (
                        dir_content[6:].strip()
                        if dir_content.startswith("elseif")
                        else dir_content[4:].strip()
                    )
                    cond_tree = parse_expression(expr_str)
                    current_cond_met = eval_tree(cond_tree, env).to_bool()
                elif depth == 1 and dir_content == "else":
                    branches.append((current_cond_met, current_branch_lines))
                    current_branch_lines = []
                    current_cond_met = True
                else:
                    current_branch_lines.append((line_no, line))
            else:
                current_branch_lines.append((line_no, line))
            i += 1
        else:
            raise MacroSyntaxError(
                "Unmatched @#if directive (missing @#endif)",
                line=first_line_no,
                filename=filename,
            )

        for cond, blines in branches:
            if cond:
                return blines, i

        return [], i


# ---------------------------------------------------------------------------
# Public Functions
# ---------------------------------------------------------------------------


def expand_macro(
    source: str,
    filepath: Optional[str] = None,
    include_paths: Optional[Sequence[str]] = None,
    defines: Optional[Dict[str, Any]] = None,
    only_if_needed: bool = False,
) -> str:
    """Expand Dynare macro processor directives in a string or file using Lark.

    Parameters
    ----------
    source : str
        Either the text content of the mod file or a file path.
    filepath : str, optional
        Path to the mod file (used for resolving includes and error reporting).
    include_paths : list of str, optional
        Directories to search for @#include directives.
    defines : dict, optional
        Pre-defined macro variables.
    only_if_needed : bool, optional
        If True and no macro directives are detected in the source, returns the
        source string exactly unchanged. Default is False.

    Returns
    -------
    str
        Expanded mod file text.
    """
    if filepath is None and os.path.isfile(source):
        filepath = source
        with open(filepath, "rt", encoding="utf-8") as f:
            content = f.read()
    else:
        content = source

    if only_if_needed and not has_macro_directives(content):
        return content

    main_dir = os.path.dirname(os.path.abspath(filepath)) if filepath else "."
    processor = MacroProcessor(
        main_dir=main_dir, include_paths=include_paths, defines=defines
    )
    return processor.expand_string(
        content, filename=filepath or "<string>", only_if_needed=False
    )


def macroexpand(
    source: str,
    filepath: Optional[str] = None,
    include_paths: Optional[Sequence[str]] = None,
    defines: Optional[Dict[str, Any]] = None,
    only_if_needed: bool = False,
) -> str:
    """Alias for expand_macro."""
    return expand_macro(
        source,
        filepath=filepath,
        include_paths=include_paths,
        defines=defines,
        only_if_needed=only_if_needed,
    )
