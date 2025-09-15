from lark.visitors import Transformer, Interpreter
from lark.tree import Tree
from lark.lexer import Token
import math
from typing import Dict, Any, Callable, Union, List
from .autodiff import DNumber as DN
import math

class DefinitionError(Exception):

    def __init__(self, msg, tree=None):

        self.msg = msg
        self.tree = tree

    def __str__(self):

        meta = self.tree.meta
        return f"({meta.line}, {meta.column}): {self.msg}"


    
class Normal:
    def __init__(self, u,v):
        self.mu = u
        self.sigma = v

class FormulaEvaluator(Interpreter):
    """
    An interpreter that evaluates mathematical formulas as defined by the grammar.
    
    This class can evaluate:
    - Basic arithmetic operations (add, sub, mul, div, pow, neg)
    - Numbers and symbols (constants, values, variables)
    - Function calls
    - Assignments and equations
    """
    
    def __init__(self, symbol_table: Dict[str, Any] = None, function_table: Dict[str, Callable] = None, steady_state=False, diff=False):
        """
        Initialize the evaluator.
        
        Args:
            symbol_table: Dictionary mapping symbol names to their values
            function_table: Dictionary mapping function names to callable functions
            steady_state: If True, evaluates variables at their steady state (only the name of the symbol is taken into account)
        """
        super().__init__()
        # self.symbol_table = symbol_table or {}
        self.function_table = function_table or {}
        self.steady_state = steady_state
        self.diff = diff

        self.constants = {}
        self.processes = {}
        self.values = {}
        self.variables = {}
        self.steady_states = {}

        self.equations = []
        self.time = None # None or integer
        self.errors = []

        # Add default mathematical functions
        from .autodiff import MATH_FUNCTIONS
        self.function_table.update(MATH_FUNCTIONS)

        self.function_table.update({'N': (lambda u,v: Normal(u,v)) })

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
        if right == 0:
            raise ZeroDivisionError("Division by zero")
        return left / right
    
    def pow(self, tree):
        """Handle exponentiation: a ^ b or a ** b"""
        base = self.visit(tree.children[0])
        exponent = self.visit(tree.children[1])
        return base ** exponent
    
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
            raise ValueError(f"({tree.meta.line},{tree.meta.column}): Undefined value: {name}")
            # self.errors.append( DefinitionError(f"Undefined constant: {name}", tree=tree) )
            # return math.nan
    
    def value(self, tree):
        """Handle values with specific time: name[time]"""
        name = str(tree.children[0].children[0])
        time = int(tree.children[1].children[0])

        # Create a key for the symbol table
        if self.steady_state:
            return self.steady_states.get(name, math.nan)
        else:
            if name not in self.values:
                self.errors.append(
                    DefinitionError(f"Undefined value {name}[~]", tree=tree)
                )
                return math.nan
            else:
                vvs = self.values[name]
                if time not in vvs:
                    self.errors.append(
                        DefinitionError(f"Undefined value {name}[{time}]", tree=tree)
                    )
                    return math.nan
                else:
                    return vvs[time]
    
    def variable(self, tree):
        """Handle variables with time indexing: name[t+shift]"""
        name = str(tree.children[0].children[0])
        index = str(tree.children[1].children[0])  # Usually 't'
        shift = int(tree.children[2].children[0])
        
        # TODO deal with index ~
        if name not in self.variables:
            self.variables[name] = {}
        
        if self.time is not None:
            time = self.time + shift
            key = f"{name}[{time}]"
            return self.values[name].get(time, math.nan)
        elif self.steady_state or (index == '~'):
            # from rich import print
            if name not in self.steady_states:
                self.errors.append(
                    DefinitionError(f"Undefined steady state for variable {name}[~]", tree=tree)
                )
            return self.steady_states.get(name, math.nan)
        else:
            return self.variables[name].get(shift, math.nan)
    
    # Function calls
    def call(self, tree):
        """Handle function calls: func_name(arg)"""
        func_name = tree.children[0].children[0].value
        args = [self.visit(c) for c in tree.children[1:]]
        
        if func_name in self.function_table:
            return self.function_table[func_name](*args)
        else:
            raise ValueError(f"Undefined function: {func_name}")
    
    # Equations and assignments
    def equality(self, tree):
        """Handle equations: left = right. Returns the difference (should be 0 for equality)"""
        left = self.visit(tree.children[0])
        right = self.visit(tree.children[1])
        return right - left  # Return difference for equation solving
    
    def assignment(self, tree):
        """Handle assignments: symbol := value or symbol <- value"""
        symbol_tree = tree.children[0]
        value = self.visit(tree.children[1])
        
        name = str(symbol_tree.children[0].children[0])
        
        if symbol_tree.data == "constant":
            key = name
            if name in self.constants:
                print(f"Warning: constant {name} redefined")
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
            

            if index=='~':
                key = f"{name}[~]"
                # self.symbol_table[key] = value
                if name not in self.steady_states:
                    self.steady_states[name] = value
            else:
                assert shift == 0
                if name in self.processes:
                    raise Exception(f"Warning: invalid redefinition of process {name}.")
                else:
                    self.processes[name] = value
                    self.steady_states[name] = value.mu

        return value
    
    def quantified_assignment(self, tree):

        bounds = tree.children[0]
        assert(bounds.data == 't_double_bound')
        lower = self.visit(bounds.children[0])
        upper = self.visit(bounds.children[1])
        try:
            assert( isinstance(lower,int) and isinstance(upper, int) and lower<upper)
        except:
            raise ValueError(f"Invalid bounds in quantified assignment: {lower}, {upper}")
        dates = range(lower, upper)
        
        symbol_tree = tree.children[1]
        name = str(symbol_tree.children[0].children[0])

        if name not in self.values:
            self.values[name] = {}

        for d in dates:
            
            self.time = d
            self.constants['t'] = d

            value = self.visit(tree.children[2])
            
            name = str(symbol_tree.children[0].children[0])
            index = str(symbol_tree.children[1].children[0])
            shift = int(symbol_tree.children[2].children[0])
            assert index=='t' and shift==0

            # self.symbol_table[key] = value
            self.values[name][d] = value
            

        # self.time = original_time
        self.constants.pop('t', None)
        self.time=None
        


    # Block handling
    def assignment_block(self, tree):
        """Handle a block of assignments"""
        results = []
        for child in tree.children:
            if hasattr(child, 'data'):  # Skip newlines
                result = self.visit(child)
                results.append(result)
        return results
    
    def equation_block(self, tree):
        """Handle a block of equations"""
        results = []
        for child in tree.children:
            if hasattr(child, 'data'):  # Skip newlines
                result = self.visit(child)
                results.append(result)
        return results
    
    def free_block(self, tree):
        """Handle a mixed block of equations and assignments"""
        results = []
        for i,child in enumerate(tree.children):
            if hasattr(child, 'data'):  # Skip newlines
                if child.data in ('equality', 'formula'):
                    # result = self.visit(child)
                    # results.append(result)
                    self.equations.append(child)
                else:
                    self.visit(child)
                # results.append(result)
        return results

# class EvalEquations(FormulaEvaluator):

#     # Equations and assignments
#     def equality(self, tree):
#         """Handle equations: left = right. Returns the difference (should be 0 for equality)"""
#         left = self.visit(tree.children[0])
#         right = self.visit(tree.children[1])
#         return right - left  # Return difference for equation solving