"""LaTeX output transformer for dyno expressions.

This module provides LatexTransformer, a Lark Transformer that converts
parse trees produced by the grammar in `grammars/grammar.lark` into LaTeX
expressions. It also provides a convenience function `latex` that accepts
either a Lark Tree (or Token) or a source string and returns the LaTeX
representation.
"""
from lark import Transformer, Tree, Token
from typing import Any


class LatexTransformer(Transformer):
    """Transform a parsed Lark tree into a LaTeX string.

    Methods return strings for every subtree so combination is simple.
    """

    def name(self, children):
        # children: [Token(NAME,...)]
        name_latex = self._greek(children[0].value)

        return name_latex

    def number(self, children):
        return children[0].value

    def time(self, children):
        # time: SIGNED_INT -> time
        return children[0].value

    def index(self, children):
        # time index: 't' or '~'
        return children[0].value

    def shift(self, children):
        # shift: may be a token like '+1' or '-2' or the TimeFixer turned it into '0'
        if not children:
            return "0"
        v = children[0]
        if isinstance(v, Token):
            return v.value
        # if already a string
        return str(v)

    def cname(self, children):
        # alias for name in some parse trees (keeps compatibility)
        return self.name(children)

    def constant(self, children):
        # children: [name]
        
        return children[0]

    def value(self, children):
        # children: [name, time]
        name = children[0]
        name_latex = self._greek(name)

        time = children[1]
        return f"{name_latex}_{{{time}}}"

    def variable(self, children):
        # children: [name, index, shift]
        name = children[0]
        index = children[1]
        shift = children[2]

        # map greek letter names to LaTeX commands for variables
        name_latex = self._greek(name)

        # shift may be like '+1' or '-2' or '0'
        try:
            s = int(str(shift))
        except Exception:
            # strip leading '+' if present
            sval = str(shift)
            if sval.startswith("+"):
                sval = sval[1:]
            try:
                s = int(sval)
            except Exception:
                s = 0

        if s == 0:
            return f"{name_latex}_{{{index}}}"
        elif s > 0:
            return f"{name_latex}_{{{index}+{s}}}"
        else:
            return f"{name_latex}_{{{index}{s}}}"

    def symbol(self, children):
        return children[0]

    def add(self, children):
        a, b = children[0], children[1]
        return f"{a} + {b}"

    def sub(self, children):
        a, b = children[0], children[1]
        return f"{a} - {b}"

    def mul(self, children):
        a, b = children[0], children[1]
        # use \cdot to make multiplication explicit in LaTeX
        return f"{a} \cdot {b}"

    def div(self, children):
        a, b = children[0], children[1]
        return f"\\frac{{{a}}}{{{b}}}"

    def pow(self, children):
        base, exponent = children[0], children[1]
        # use braces around exponent
        return f"{{{base}}}^{{{exponent}}}"

    def neg(self, children):
        a = children[0]
        return f"-\left({a}\right)"

    def call(self, children):
        # children: [funname, arg1, arg2, ...]
        fun = children[0]
        args = children[1:]
        if args:
            joined = ", ".join(args)
        else:
            joined = ""
        # render function name in \mathrm to avoid italic math font
        return f"\\mathrm{{{fun}}}\left({joined}\\right)"

    def number(self, children):
        return children[0].value

    def equality(self, children):
        a, b = children[0], children[1]
        return f"{a} = {b}"

    def assignment(self, children):
        # assignment uses := or <- in source, render as '=' in LaTeX
        a, b = children[0], children[1]
        return f"{a} = {b}"

    def inequality(self, children):
        # children: [left, op_token, right]
        left = children[0]
        op = children[1]
        right = children[2]
        if isinstance(op, Token):
            op = op.value
        return f"{left} {op} {right}"

    def double_bound(self, children):
        # robust handler if encountered: a <= b <= c
        a, op1, b, op2, c = children
        return f"{a} {op1} {b} {op2} {c}"

    def call_arg(self, children):
        # fallback if grammar produces grouped args
        return children

    # Fallback for any tree we didn't explicitly handle: join children
    def __default__(self, data, children, meta):
        # If children are already strings, join them sensibly
        try:
            return " ".join(children)
        except Exception:
            return str(data)

    def _greek(self, name: str) -> str:
        """If `name` is a Greek letter name, return the corresponding
        LaTeX command (with leading backslash). Otherwise return `name`.
        Handles common lowercase names; for capitalized names returns the
        corresponding LaTeX uppercase command when available.
        """
        low = name.lower()
        greek_lower = {
            "alpha": "\\alpha",
            "beta": "\\beta",
            "gamma": "\\gamma",
            "delta": "\\delta",
            "epsilon": "\\epsilon",
            "zeta": "\\zeta",
            "eta": "\\eta",
            "theta": "\\theta",
            "iota": "\\iota",
            "kappa": "\\kappa",
            "lambda": "\\lambda",
            "mu": "\\mu",
            "nu": "\\nu",
            "xi": "\\xi",
            "omicron": "\\mathrm{o}",
            "pi": "\\pi",
            "rho": "\\rho",
            "sigma": "\\sigma",
            "tau": "\\tau",
            "upsilon": "\\upsilon",
            "phi": "\\phi",
            "chi": "\\chi",
            "psi": "\\psi",
            "omega": "\\omega",
            # common alternative names
            "varepsilon": "\\varepsilon",
            "varphi": "\\varphi",
            "varsigma": "\\varsigma",
        }
        # Uppercase LaTeX commands exist only for a subset of greek letters
        greek_upper = {
            "gamma": "\\Gamma",
            "delta": "\\Delta",
            "theta": "\\Theta",
            "lambda": "\\Lambda",
            "xi": "\\Xi",
            "pi": "\\Pi",
            "sigma": "\\Sigma",
            "upsilon": "\\Upsilon",
            "phi": "\\Phi",
            "psi": "\\Psi",
            "omega": "\\Omega",
        }

        if low in greek_lower:
            if name.islower():
                return greek_lower[low]
            # capitalized form requested
            return greek_upper.get(low, name)
        return name


def latex(expr: Any) -> str:
    """Return the LaTeX representation for `expr`.

    `expr` can be:
    - a Lark Tree (result of parsing with the project's parser), or
    - a plain string: in that case this function attempts to import the
      project's parser and parse `expr` as a `formula`.

    The function returns a string containing a LaTeX math expression.
    """
    # Avoid a hard import cycle when used from other modules: import parser
    # only when needed.
    if isinstance(expr, (Tree, Token)):
        return LatexTransformer().transform(expr)

    # assume a source string: import the project parser to build a tree
    try:
        from dyno.dynsym.grammar import parser
    except Exception:
        # best-effort: if parser is not importable, raise a clear error
        raise RuntimeError("Cannot import project parser to parse source string")

    tree = parser.parse(expr, start="formula")
    return LatexTransformer().transform(tree)


__all__ = ["LatexTransformer", "latex"]