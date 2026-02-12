import math
import numpy as np


def _get_math_module(x):
    """Return the appropriate math module (math or numpy) based on input type."""
    if isinstance(x, np.ndarray):
        return np
    else:
        return math


class DNumber:

    def __init__(self, value, derivatives=None):
        self.value = value
        self.derivatives = derivatives if derivatives is not None else {}

    def __add__(self, other):
        if isinstance(other, DNumber):
            new_value = self.value + other.value
            new_derivatives = self.derivatives.copy()
            for var, deriv in other.derivatives.items():
                if var in new_derivatives:
                    new_derivatives[var] += deriv
                else:
                    new_derivatives[var] = deriv
            return DNumber(new_value, new_derivatives)
        else:
            return DNumber(self.value + other, self.derivatives.copy())

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, DNumber):
            new_value = self.value - other.value
            new_derivatives = self.derivatives.copy()
            for var, deriv in other.derivatives.items():
                if var in new_derivatives:
                    new_derivatives[var] -= deriv
                else:
                    new_derivatives[var] = -deriv
            return DNumber(new_value, new_derivatives)
        else:
            return DNumber(self.value - other, self.derivatives.copy())

    def __rsub__(self, other):
        if isinstance(other, DNumber):
            return other.__sub__(self)
        else:
            new_derivatives = {var: -deriv for var, deriv in self.derivatives.items()}
            return DNumber(other - self.value, new_derivatives)

    def __mul__(self, other):
        if isinstance(other, DNumber):
            new_value = self.value * other.value
            new_derivatives = {}
            for var in set(self.derivatives.keys()).union(other.derivatives.keys()):
                deriv1 = self.derivatives.get(var, 0)
                deriv2 = other.derivatives.get(var, 0)
                new_derivatives[var] = deriv1 * other.value + deriv2 * self.value
            return DNumber(new_value, new_derivatives)
        else:
            new_derivatives = {
                var: deriv * other for var, deriv in self.derivatives.items()
            }
            return DNumber(self.value * other, new_derivatives)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        if isinstance(other, DNumber):
            new_value = self.value / other.value
            new_derivatives = {}
            for var in set(self.derivatives.keys()).union(other.derivatives.keys()):
                deriv1 = self.derivatives.get(var, 0)
                deriv2 = other.derivatives.get(var, 0)
                new_derivatives[var] = (deriv1 * other.value - deriv2 * self.value) / (
                    other.value**2
                )
            return DNumber(new_value, new_derivatives)
        else:
            new_derivatives = {
                var: deriv / other for var, deriv in self.derivatives.items()
            }
            return DNumber(self.value / other, new_derivatives)

    def __rtruediv__(self, other):

        if isinstance(other, DNumber):
            return other.__truediv__(self)
        else:
            new_derivatives = {}
            for var, deriv in self.derivatives.items():
                new_derivatives[var] = -deriv * other / (self.value**2)
            return DNumber(other / self.value, new_derivatives)

    def __pow__(self, power):

        if isinstance(power, DNumber):
            new_value = self.value**power.value
            new_derivatives = {}
            m = _get_math_module(self.value)
            for var in set(self.derivatives.keys()).union(power.derivatives.keys()):
                deriv1 = self.derivatives.get(var, 0)
                deriv2 = power.derivatives.get(var, 0)
                new_derivatives[var] = new_value * (
                    deriv1 * power.value / self.value + deriv2 * m.log(self.value)
                )
            return DNumber(new_value, new_derivatives)
        else:
            new_value = self.value**power
            new_derivatives = {
                var: deriv * power * (self.value ** (power - 1))
                for var, deriv in self.derivatives.items()
            }
            return DNumber(new_value, new_derivatives)

    def __rpow__(self, base):
        if isinstance(base, DNumber):
            return base.__pow__(self)
        else:
            new_value = base**self.value
            m = _get_math_module(self.value)
            new_derivatives = {
                var: deriv * new_value * m.log(base)
                for var, deriv in self.derivatives.items()
            }
            return DNumber(new_value, new_derivatives)

    def __neg__(self):
        new_derivatives = {var: -deriv for var, deriv in self.derivatives.items()}
        return DNumber(-self.value, new_derivatives)

    def __repr__(self):
        return f"DNumber(value={self.value}, derivatives={self.derivatives})"


# Math functions for dual numbers


def sin(x):
    """Sine function that works with both floats and DNumber objects."""
    if isinstance(x, DNumber):
        new_value = sin(x.value)
        new_derivatives = {
            var: deriv * cos(x.value) for var, deriv in x.derivatives.items()
        }
        return DNumber(new_value, new_derivatives)
    else:
        m = _get_math_module(x)
        return m.sin(x)


def cos(x):
    """Cosine function that works with both floats and DNumber objects."""
    if isinstance(x, DNumber):
        new_value = cos(x.value)
        new_derivatives = {
            var: -deriv * sin(x.value) for var, deriv in x.derivatives.items()
        }
        return DNumber(new_value, new_derivatives)
    else:
        m = _get_math_module(x)
        return m.cos(x)


def tan(x):
    """Tangent function that works with both floats and DNumber objects."""
    if isinstance(x, DNumber):
        new_value = tan(x.value)
        sec_squared = 1 / (cos(x.value) ** 2)
        new_derivatives = {
            var: deriv * sec_squared for var, deriv in x.derivatives.items()
        }
        return DNumber(new_value, new_derivatives)
    else:
        m = _get_math_module(x)
        return m.tan(x)


def exp(x):
    """Exponential function that works with both floats and DNumber objects."""
    from rich import print

    if isinstance(x, DNumber):
        new_value = exp(x.value)
        new_derivatives = {
            var: deriv * new_value for var, deriv in x.derivatives.items()
        }
        return DNumber(new_value, new_derivatives)
    else:
        m = _get_math_module(x)
        return m.exp(x)


def log(x):
    """Natural logarithm function that works with both floats and DNumber objects."""
    if isinstance(x, DNumber):
        new_value = log(x.value)
        new_derivatives = {var: deriv / x.value for var, deriv in x.derivatives.items()}
        return DNumber(new_value, new_derivatives)
    else:
        m = _get_math_module(x)
        return m.log(x)


def sqrt(x):
    """Square root function that works with both floats and DNumber objects."""
    if isinstance(x, DNumber):
        new_value = sqrt(x.value)
        new_derivatives = {
            var: deriv / (2 * new_value) for var, deriv in x.derivatives.items()
        }
        return DNumber(new_value, new_derivatives)
    else:
        m = _get_math_module(x)
        return m.sqrt(x)


def dabs(x):
    """Absolute value function that works with both floats and DNumber objects."""
    if isinstance(x, DNumber):
        new_value = abs(x.value)
        sign = 1 if x.value >= 0 else -1
        new_derivatives = {var: deriv * sign for var, deriv in x.derivatives.items()}
        return DNumber(new_value, new_derivatives)
    else:
        return abs(x)


def sinh(x):
    """Hyperbolic sine function that works with both floats and DNumber objects."""
    if isinstance(x, DNumber):
        new_value = sinh(x.value)
        new_derivatives = {
            var: deriv * cosh(x.value) for var, deriv in x.derivatives.items()
        }
        return DNumber(new_value, new_derivatives)
    else:
        m = _get_math_module(x)
        return m.sinh(x)


def cosh(x):
    """Hyperbolic cosine function that works with both floats and DNumber objects."""
    if isinstance(x, DNumber):
        new_value = cosh(x.value)
        new_derivatives = {
            var: deriv * sinh(x.value) for var, deriv in x.derivatives.items()
        }
        return DNumber(new_value, new_derivatives)
    else:
        m = _get_math_module(x)
        return m.cosh(x)


def tanh(x):
    """Hyperbolic tangent function that works with both floats and DNumber objects."""
    if isinstance(x, DNumber):
        new_value = tanh(x.value)
        sech_squared = 1 - new_value**2
        new_derivatives = {
            var: deriv * sech_squared for var, deriv in x.derivatives.items()
        }
        return DNumber(new_value, new_derivatives)
    else:
        m = _get_math_module(x)
        return m.tanh(x)


def asin(x):
    """Arcsine function that works with both floats and DNumber objects."""
    if isinstance(x, DNumber):
        new_value = asin(x.value)
        derivative_factor = 1 / sqrt(1 - x.value**2)
        new_derivatives = {
            var: deriv * derivative_factor for var, deriv in x.derivatives.items()
        }
        return DNumber(new_value, new_derivatives)
    else:
        m = _get_math_module(x)
        return np.arcsin(x) if m is np else math.asin(x)


def acos(x):
    """Arccosine function that works with both floats and DNumber objects."""
    if isinstance(x, DNumber):
        new_value = acos(x.value)
        derivative_factor = -1 / sqrt(1 - x.value**2)
        new_derivatives = {
            var: deriv * derivative_factor for var, deriv in x.derivatives.items()
        }
        return DNumber(new_value, new_derivatives)
    else:
        m = _get_math_module(x)
        return np.arccos(x) if m is np else math.acos(x)


def atan(x):
    """Arctangent function that works with both floats and DNumber objects."""
    if isinstance(x, DNumber):
        new_value = atan(x.value)
        derivative_factor = 1 / (1 + x.value**2)
        new_derivatives = {
            var: deriv * derivative_factor for var, deriv in x.derivatives.items()
        }
        return DNumber(new_value, new_derivatives)
    else:
        m = _get_math_module(x)
        return np.arctan(x) if m is np else math.atan(x)


def dmax(x, y):
    """Maximum function that works with both floats and DNumber objects."""
    if isinstance(x, DNumber) or isinstance(y, DNumber):
        # Convert to DNumber if needed
        if not isinstance(x, DNumber):
            x = DNumber(x)
        if not isinstance(y, DNumber):
            y = DNumber(y)

        if x.value >= y.value:
            return x
        else:
            return y
    else:
        return max(x, y)


def dmin(x, y):
    """Minimum function that works with both floats and DNumber objects."""
    if isinstance(x, DNumber) or isinstance(y, DNumber):
        # Convert to DNumber if needed
        if not isinstance(x, DNumber):
            x = DNumber(x)
        if not isinstance(y, DNumber):
            y = DNumber(y)

        if x.value <= y.value:
            return x
        else:
            return y
    else:
        return min(x, y)


def log10(x):
    """Base-10 logarithm function that works with both floats and DNumber objects."""
    if isinstance(x, DNumber):
        new_value = log10(x.value)
        new_derivatives = {
            var: deriv / (x.value * log(10)) for var, deriv in x.derivatives.items()
        }
        return DNumber(new_value, new_derivatives)
    else:
        m = _get_math_module(x)
        return m.log10(x)


def log2(x):
    """Base-2 logarithm function that works with both floats and DNumber objects."""
    if isinstance(x, DNumber):
        new_value = log2(x.value)
        new_derivatives = {
            var: deriv / (x.value * log(2)) for var, deriv in x.derivatives.items()
        }
        return DNumber(new_value, new_derivatives)
    else:
        m = _get_math_module(x)
        return m.log2(x)


def floor(x):
    """Floor function that works with both floats and DNumber objects."""
    if isinstance(x, DNumber):
        new_value = floor(x.value)
        # Derivative of floor is 0 everywhere except at integer points (where it's undefined)
        new_derivatives = {var: 0.0 for var in x.derivatives.keys()}
        return DNumber(new_value, new_derivatives)
    else:
        m = _get_math_module(x)
        return m.floor(x)


def ceil(x):
    """Ceiling function that works with both floats and DNumber objects."""
    if isinstance(x, DNumber):
        new_value = ceil(x.value)
        # Derivative of ceil is 0 everywhere except at integer points (where it's undefined)
        new_derivatives = {var: 0.0 for var in x.derivatives.keys()}
        return DNumber(new_value, new_derivatives)
    else:
        m = _get_math_module(x)
        return m.ceil(x)


def pow(x, y):
    """Power function that works with both floats and DNumber objects."""
    if isinstance(x, DNumber) or isinstance(y, DNumber):
        if isinstance(x, DNumber):
            return x.__pow__(y)
        else:
            # x is float, y is DNumber
            if not isinstance(y, DNumber):
                y = DNumber(y)
            return DNumber(x).__rpow__(y)
    else:
        return x**y


# Dictionary mapping function names to dual number implementations
MATH_FUNCTIONS = {
    "sin": sin,
    "cos": cos,
    "tan": tan,
    "exp": exp,
    "log": log,
    "sqrt": sqrt,
    "abs": dabs,
    "sinh": sinh,
    "cosh": cosh,
    "tanh": tanh,
    "asin": asin,
    "acos": acos,
    "atan": atan,
    "max": dmax,
    "min": dmin,
    "log10": log10,
    "log2": log2,
    "floor": floor,
    "ceil": ceil,
    "pow": pow,
}
