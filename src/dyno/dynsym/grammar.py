from hashlib import new
from lark.exceptions import (
    LarkError,
    UnexpectedInput,
    ConfigurationError,
    UnexpectedCharacters,
)
from yaml import ScalarNode

import copy

# import lark
from lark import Lark
from lark.visitors import v_args
from lark.tree import Tree
from lark.lexer import Token
from lark.visitors import Interpreter, Visitor, Transformer
from os import path

from typing import Tuple, Dict, Set, Union, List


GRAMMARS_PATH = path.join(
    path.split(__file__)[0],
    "grammars/"
)
GRAMMAR_FILE = path.join(GRAMMARS_PATH, 'grammar.lark')

grammar_0 = open(GRAMMAR_FILE, "rt", encoding="utf-8").read()


### replaces date with 0 when missing

class TimeFixer(Transformer):

    @v_args(tree=True)
    def shift(self, tree):

        if tree.children[0] is None:
            return Tree(
                "shift",
                ["0"]
            )
        else:
            return tree


parser = Lark(
    grammar_0,
    start=[
        "formula",
        "equation_block",
        "assignment_block",
        "free_block"
    ],
    parser="lalr",
    strict=True,
    propagate_positions=True,
    transformer=TimeFixer()
)


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
        
        if shift<0:
            return f"{name}[{index}{shift}]"
        elif shift>0:
            return f"{name}[{index}+{shift}]"
        else:
            return f"{name}[{index}]"


    def equality(self, tree):
        a = self.visit(tree.children[0])
        b = self.visit(tree.children[1])
        return f"{a} = {b}"

    def double_complementarity(self, tree):
        a = self.visit(tree.children[0])
        b = self.visit(tree.children[1])
        return f"{a} ⟂ {b}"

    def double_inequality(self, tree):
        a = self.visit(tree.children[0])
        b = self.visit(tree.children[1])
        c = self.visit(tree.children[2])
        return f"{a} <= {b} <= {c}"

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
        funname = tree.children[0].children[0].value
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
    return Printer(stringify_symbols=stringify_symbols).visit(expr)


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
    time = arg[1]

    if time < 0:
        return "{}__m{}_".format(s, str(-time))
    else:
        return "{}__p{}_".format(s, str(time))


def stringify_variable(arg: Tuple[str, Tuple[str, int]]) -> str:
    s = arg[0]
    time = arg[1][0]
    shift = int(arg[1][1])

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
    @wraps(f)
    def wrapper(*args, **kwds):
        if not isinstance(args[0], str):
            return f(*args, **kwds)
        else:
            a = parser.parse(args[0], start="start")
            nargs = tuple([a]) + args[1:]
            res = f(*nargs, **kwds)
            return str_expression(res)

    return wrapper



## these functions apply the vistors/transformers either on expressions or on strings
# @expression_or_string
def stringify(expr: Expression)->str:
    return Stringifier().transform(expr)

