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
    
    # Operator precedence (higher number = higher precedence)
    _PRECEDENCE = {
        'add': 1, 'sub': 1,
        'mul': 2, 'div': 2,
        'pow': 3,
        'neg': 4,
        'atom': 5  # highest precedence for atoms
    }

    def name(self, children):
        # children: [Token(NAME,...)]
        name_latex = self._greek(str(children[0]))
        return self._with_precedence(name_latex, 'atom')

    def number(self, children):
        return self._with_precedence(children[0].value, 'atom')

    def time(self, children):
        # time: SIGNED_INT -> time
        return str(children[0])

    def index(self, children):
        # time index: 't' or '~'
        return str(children[0])

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
        name_str = self._get_string(children[0])
        return self._with_precedence(name_str, 'atom')

    def value(self, children):
        # children: [name, time]
        name = self._get_string(children[0])
        name_latex = self._greek(name)

        time = children[1]
        result = f"{name_latex}_{{{time}}}"
        return self._with_precedence(result, 'atom')

    def variable(self, children):
        # children: [name, index, shift]
        name = self._get_string(children[0])
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
            result = f"{name_latex}_{{{index}}}"
        elif s > 0:
            result = f"{name_latex}_{{{index}+{s}}}"
        else:
            result = f"{name_latex}_{{{index}{s}}}"
        
        return self._with_precedence(result, 'atom')

    def symbol(self, children):
        return self._with_precedence(children[0], 'atom')

    def add(self, children):
        a, b = children[0], children[1]
        # Get the raw strings without precedence info
        a_str = self._get_string(a)
        b_str = self._get_string(b)
        result = f"{a_str} + {b_str}"
        return self._with_precedence(result, 'add')

    def sub(self, children):
        a, b = children[0], children[1]
        a_str = self._get_string(a)
        # For subtraction, right operand needs parentheses if it has lower precedence
        b_str = self._get_string_with_parens(b, 'sub', 'right')
        result = f"{a_str} - {b_str}"
        return self._with_precedence(result, 'sub')

    def mul(self, children):
        a, b = children[0], children[1]
        # Both operands need parentheses if they have lower precedence than mul
        a_str = self._get_string_with_parens(a, 'mul', 'left')
        b_str = self._get_string_with_parens(b, 'mul', 'right')
        # use \cdot to make multiplication explicit in LaTeX
        result = f"{a_str} \\cdot {b_str}"
        return self._with_precedence(result, 'mul')

    def div(self, children):
        a, b = children[0], children[1]
        a_str = self._get_string(a)
        b_str = self._get_string(b)
        # Division uses \frac, so no need for parentheses around operands
        result = f"\\frac{{{a_str}}}{{{b_str}}}"
        return self._with_precedence(result, 'div')

    def pow(self, children):
        base, exponent = children[0], children[1]
        # Base needs parentheses if it has lower precedence than pow
        base_str = self._get_string_with_parens(base, 'pow', 'left')
        exp_str = self._get_string(exponent)
        # use braces around base and exponent
        result = f"{{{base_str}}}^{{{exp_str}}}"
        return self._with_precedence(result, 'pow')

    def neg(self, children):
        a = children[0]
        a_str = self._get_string(a)
        # Only add parentheses if the operand is a complex expression (not an atom)
        if self._get_precedence(a) < self._PRECEDENCE['atom']:
            result = f"-({a_str})"
        else:
            result = f"-{a_str}"
        return self._with_precedence(result, 'neg')

    def call(self, children):
        # children: [funname, arg1, arg2, ...]
        fun = self._get_string(children[0])
        args = [self._get_string(arg) for arg in children[1:]]
        if args:
            joined = ", ".join(args)
        else:
            joined = ""
        # render function name in \mathrm to avoid italic math font
        result = f"\\mathrm{{{fun}}}\\left({joined}\\right)"
        return self._with_precedence(result, 'atom')



    def equality(self, children):
        a = self._get_string(children[0])
        b = self._get_string(children[1])
        result = f"{a} = {b}"
        return self._with_precedence(result, 'atom')

    def assignment(self, children):
        # assignment uses := or <- in source, render as '=' in LaTeX
        a = self._get_string(children[0])
        b = self._get_string(children[1])
        result = f"{a} = {b}"
        return self._with_precedence(result, 'atom')

    def inequality(self, children):
        # children: [left, op_token, right]
        left = self._get_string(children[0])
        op = children[1]
        right = self._get_string(children[2])
        if isinstance(op, Token):
            op = op.value
        result = f"{left} {op} {right}"
        return self._with_precedence(result, 'atom')

    def double_bound(self, children):
        # robust handler if encountered: a <= b <= c
        a = self._get_string(children[0])
        op1 = children[1]
        b = self._get_string(children[2])
        op2 = children[3]
        c = self._get_string(children[4])
        result = f"{a} {op1} {b} {op2} {c}"
        return self._with_precedence(result, 'atom')

    def call_arg(self, children):
        # fallback if grammar produces grouped args
        return children

    # Fallback for any tree we didn't explicitly handle: join children
    def __default__(self, data, children, meta):
        # If children are already strings, join them sensibly
        try:
            joined = " ".join(self._get_string(child) for child in children)
            return self._with_precedence(joined, 'atom')
        except Exception:
            return self._with_precedence(str(data), 'atom')

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
            "khi": "\\chi",
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

    def _with_precedence(self, latex_str: str, op_type: str):
        """Attach precedence information to a LaTeX string."""
        return (latex_str, self._PRECEDENCE[op_type])
    
    def _get_string(self, item):
        """Extract the LaTeX string from an item, ignoring precedence."""
        if isinstance(item, tuple):
            return item[0]
        return str(item)
    
    def _get_precedence(self, item):
        """Extract the precedence from an item."""
        if isinstance(item, tuple):
            return item[1]
        return self._PRECEDENCE['atom']  # assume atoms have highest precedence
    
    def _get_string_with_parens(self, item, parent_op: str, position: str):
        """Get string with parentheses if needed based on operator precedence."""
        item_str = self._get_string(item)
        item_prec = self._get_precedence(item)
        parent_prec = self._PRECEDENCE[parent_op]
        
        # Add parentheses if:
        # 1. Child has lower precedence than parent
        # 2. For right-associative operations and equal precedence on the right
        needs_parens = False
        
        if item_prec < parent_prec:
            needs_parens = True
        elif item_prec == parent_prec:
            # Special cases for equal precedence
            if parent_op in ['sub', 'div'] and position == 'right':
                needs_parens = True
            # For pow, right operand doesn't need parens: a^b^c = a^(b^c)
            elif parent_op == 'pow' and position == 'left':
                needs_parens = True
        
        if needs_parens:
            return f"\\left({item_str}\\right)"
        return item_str


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
        result = LatexTransformer().transform(expr)
        # Extract string from precedence tuple if needed
        if isinstance(result, tuple):
            return result[0]
        return str(result)

    # assume a source string: import the project parser to build a tree
    try:
        from dyno.dynsym.grammar import parser
    except Exception:
        # best-effort: if parser is not importable, raise a clear error
        raise RuntimeError("Cannot import project parser to parse source string")

    tree = parser.parse(expr, start="formula")
    result = LatexTransformer().transform(tree)
    # Extract string from precedence tuple if needed
    if isinstance(result, tuple):
        return result[0]
    return str(result)


__all__ = ["LatexTransformer", "latex"]