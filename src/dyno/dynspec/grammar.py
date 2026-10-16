from hashlib import new
from lark.exceptions import (
    LarkError,
    UnexpectedInput,
    ConfigurationError,
    UnexpectedCharacters,
)
from yaml import ScalarNode

import copy
import functools

# import lark
from lark import Lark
from lark.visitors import v_args
from lark.tree import Tree
from lark.lexer import Token
from lark.visitors import Interpreter, Visitor, Transformer
from os import path

from typing import Tuple, Dict, Set, Union, List, cast

GRAMMARS_PATH = path.join(path.split(__file__)[0], "grammars/")
GRAMMAR_FILE = path.join(GRAMMARS_PATH, "grammar.lark")

grammar_0 = open(GRAMMAR_FILE, "rt", encoding="utf-8").read()


### replaces date with 0 when missing


class TimeFixer(Transformer):

    @v_args(tree=True)
    def shift(self, tree):

        if tree.children[0] is None:
            return Tree("shift", ["0"])
        else:
            return tree

    @v_args(tree=True)
    def complementarity(self, tree):
        """Normalize `res ⟂ a <= b [<= c]` to `complementarity(res, lb, x, ub)`.

        A missing bound is made explicit as `-inf` or `inf`. In the one-sided form,
        the bounded variable is whichever side of `<=` is a variable; if both are,
        it is the one dated `[t]`.
        """
        res, a, b, c = tree.children
        if c is not None:
            lb, x, ub = a, b, c
            if not _is_variable(x):
                raise _complementarity_error(
                    "the middle term of `lb <= x <= ub` must be a variable", tree
                )
        else:
            if _is_variable(a) and _is_variable(b):
                if _is_current(a) == _is_current(b):
                    raise _complementarity_error(
                        "ambiguous one-sided bound: both sides of `<=` are variables; "
                        "write `lb <= x <= ub` explicitly",
                        tree,
                    )
                bounded_is_rhs = _is_current(b)
            elif _is_variable(a) or _is_variable(b):
                bounded_is_rhs = _is_variable(b)
            else:
                raise _complementarity_error(
                    "the constraint must bound a variable (`lb <= x[t]` or `x[t] <= ub`)",
                    tree,
                )
            if bounded_is_rhs:
                lb, x, ub = a, b, _infinity(tree.meta)
            else:
                lb, x, ub = Tree("neg", [_infinity(tree.meta)], tree.meta), a, b
        return Tree("complementarity", [res, lb, x, ub], tree.meta)


def _is_variable(tree) -> bool:
    return isinstance(tree, Tree) and tree.data == "variable"


def _is_current(tree: Tree) -> bool:
    return int(str(tree.children[2].children[0])) == 0


def _infinity(meta) -> Tree:
    return Tree("constant", [Tree("name", [Token("NAME", "inf")])], meta)


def _complementarity_error(message: str, tree: Tree):
    from dyno.errors import DefinitionError

    # Positions are only propagated to `tree` after the inline transformer runs,
    # so locate the error on the first child that has them (the residual).
    located = next(
        (c for c in tree.children if isinstance(c, Tree) and not c.meta.empty), tree
    )
    return DefinitionError(
        f"Invalid complementarity condition: {message}", tree=located
    )


@functools.cache
def get_parser() -> Lark:
    """Build (once) and return the dyno grammar parser.

    Building is deferred to first use to keep ``import dyno`` fast; Lark's
    on-disk cache makes subsequent builds (in new processes) nearly free.
    """
    return Lark(
        grammar_0,
        start=["formula", "equation_block", "assignment_block", "free_block"],
        parser="lalr",
        strict=True,
        propagate_positions=True,
        transformer=TimeFixer(),
        cache=True,
    )


class _LazyParser:
    """Proxy forwarding attribute access to the parser built by `get_parser`."""

    def __getattr__(self, name):
        return getattr(get_parser(), name)


parser = cast(Lark, _LazyParser())


Expression = Union[Tree, Token]


# Prints a tree as a string
# WIP!!! (probably correct, but way too many parentheses)
class Printer(Interpreter):

    def __init__(self, stringify_symbols=False):
        self.stringify_symbols = stringify_symbols
        super().__init__()

    def add(self, tree):
        if len(tree.children) == 1:
            return "ERROR"
        a = self.visit(tree.children[0])
        b = self.visit(tree.children[1])
        return f"{a} + {b}"

    def sub(self, tree):

        if len(tree.children) == 1:
            return "ERROR"
        a = self.visit(tree.children[0])
        b = self.visit(tree.children[1])
        return f"{a} - ({b})"

    def constant(self, tree):

        name = str(tree.children[0].children[0])

        if self.stringify_symbols:
            s = stringify_constant(name)
        else:
            s = name
        return s

    def value(self, tree):

        name = str(tree.children[0].children[0])
        date = int(tree.children[1].children[0])

        if self.stringify_symbols:
            s = stringify_value((name, date))
        else:
            s = f"{name}[{date}]"

        return s

    def variable(self, tree):

        name = str(tree.children[0].children[0])
        index = str(tree.children[1].children[0])
        shift = int(tree.children[2].children[0])

        if self.stringify_symbols:
            s = stringify_variable((name, (index, shift)))
            return s

        if shift < 0:
            return f"{name}[{index}{shift}]"
        elif shift > 0:
            return f"{name}[{index}+{shift}]"
        else:
            return f"{name}[{index}]"

    def equality(self, tree):
        a = self.visit(tree.children[0])
        b = self.visit(tree.children[1])
        return f"{a} = {b}"

    def complementarity(self, tree):
        res, lb, x, ub = (self.visit(c) for c in tree.children)
        return f"{res} ⟂ {lb} <= {x} <= {ub}"

    def assignment(self, tree):
        a = self.visit(tree.children[0])
        b = self.visit(tree.children[1])
        return f"{a} = {b}"

    def symbol(self, tree):
        name = tree.children[0].value
        return name

    def mul(self, tree):
        a = self.visit(tree.children[0])
        b = self.visit(tree.children[1])
        return f"({a})*({b})"

    def div(self, tree):
        a = self.visit(tree.children[0])
        b = self.visit(tree.children[1])
        return f"({a})/({b})"

    def call(self, tree):

        # TODO: correct the small inconsistency below
        # between modfile lark and dyno lark
        try:
            funname = tree.children[0].children[0].value
        except (AttributeError, IndexError):
            # print(tree.pretty())
            funname = str(tree.children[0].children[0])

        args = self.visit(tree.children[1])
        return f"{funname}({args})"

    def pow(self, tree):
        arg1 = self.visit(tree.children[0])
        arg2 = self.visit(tree.children[1])
        return f"({arg1})^({arg2})"

    def number(self, tree):
        return tree.children[0].value

    def signed_int(self, tree):
        return tree.children[0].value

    def neg(self, tree):
        a = self.visit(tree.children[0])
        return f"-({a})"

    def expectation(self, tree):
        a = self.visit(tree.children[0])
        return f"𝔼[ {a} ]"

    def inequality(self, tree):
        a = self.visit(tree.children[0])
        b = (tree.children[1]).value
        c = self.visit(tree.children[2])
        return f"{a} {b} {c}"

    def predicate(self, tree):
        if len(tree.children) == 1:
            return self.visit(tree.children[0])
        else:
            return "∀t, " + self.visit(tree.children[1])


# prints expression
def str_expression(expr: Expression, stringify_symbols=False) -> str:
    return Printer(stringify_symbols=stringify_symbols).visit(cast(Tree, expr))


# that one is ambiguous because of compatibility concerns.
def create_variable(name, shift=0, time="t"):
    return Tree(
        "variable",
        [
            Tree("name", [name]),
            Tree("time", [time]),
            Tree("shift", [str(shift)]),
        ],
    )


def stringify_constant(p: str) -> str:
    return "{}_".format(p)


def stringify_value(arg: Tuple[str, int]) -> str:

    s = arg[0]
    time = int(arg[1])

    if time < 0:
        return "{}__m{}_".format(s, str(-time))
    else:
        return "{}__p{}_".format(s, str(time))


def stringify_variable(arg: Tuple[str, Tuple[str, int]]) -> str:
    s, (time, shift) = arg
    shift = int(shift)

    if shift < 0:
        return "{}__{}_m{}_".format(s, time, str(-shift))
    else:
        return "{}__{}_p{}_".format(s, time, str(shift))


def stringify_symbol(arg) -> str:

    if isinstance(arg, str):
        return stringify_constant(arg)
    elif isinstance(arg, tuple):
        if len(arg[1]) == 1:
            return stringify_value(arg)
        elif len(arg[1]) == 2:
            return stringify_variable(arg)

    raise Exception("Unknown canonical form: {}".format(arg))


# decorator to define functions which operate
# either on Trees or on strings.
def expression_or_string(f):
    @functools.wraps(f)
    def wrapper(*args, **kwds):
        if not isinstance(args[0], str):
            return f(*args, **kwds)
        else:
            a = parser.parse(args[0], start="start")
            nargs = tuple([a]) + args[1:]
            res = f(*nargs, **kwds)
            return str_expression(res)

    return wrapper
