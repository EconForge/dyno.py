import os
import time
dir_path = os.path.dirname(os.path.realpath(__file__))

modfile_grammar = open(f"src/dyno/dynsym/grammars/modfile_grammar.lark").read()
modtxt = open(f"{dir_path}/examples/modfiles/example2.mod").read()
from lark import Tree, Token, Lark
from lark.visitors import Transformer, v_args

from dyno.dynsym.analyze import FormulaEvaluator

@v_args(tree=True)
class ModFileTransformer(Transformer):

    def __init__(self):

        self.variables = []
        self.parameters = []
        self.variables_exo = []
        self.variables_pred = []
        self.equations = []

        super().__init__()

    def var_statement(self, tree):
        for child in tree.children:
            name = str(child.children[0].children[0])
            self.variables.append(name)
        return tree


    def varexo_statement(self, tree):
        for child in tree.children:
            name = str(child.children[0].children[0])
            self.variables_exo.append(name)
        return tree

    
    def par_statement(self, tree):
        for child in tree.children:
            name = str(child.children[0].children[0])
            self.parameters.append(name)
        return tree

    
    def pred_statement(self, tree):
        for child in tree.children:
            name = str(child.children[0].children[0])
            self.variables_pred.append(name)
        return tree
    
    def constant(self, tree):
        name = str(tree.children[0].children[0])

        if name in self.variables + self.variables_exo + self.variables_pred:
            return Tree(
                "variable", [
                    Tree("name" ,[name]),
                    Tree("index",['t']),
                    Tree("shift",["0"])
                ]
            )
        return tree
    
    def variable(self, tree):

        return Tree(
            "variable", [
                tree.children[0],
                Tree("index",["t"]),
                tree.children[1]
            ]
        )
    
    # def lequation(self, tree):

    #     self.equations.append(tree)
    #     return tree
                       


class InterpretModfile(FormulaEvaluator):

    def __init__(self, steady_state=False):


        super().__init__(steady_state=steady_state)

        self.steady_states = {}
        self.constants = {}
        self.covariances = {}
        self.current_block = None

    def var_statement(self, tree):
        return tree


    def varexo_statement(self, tree):
        return tree

    
    def par_statement(self, tree):
        return tree

    
    def pred_statement(self, tree):
        return tree
    
    def initval_block(self, tree):

        self.current_block = "initval"
        res = [self.visit(ch) for ch in tree.children]
        self.current_block = None
      
    def parassignment(self, tree):
        name = str(tree.children[0].children[0].children[0])
        formula = (tree.children[1])
        value = self.visit(formula)
        if self.current_block is None:
            self.constants[name] = value
        elif self.current_block == "initval":
            self.steady_states[name] = value
        # elif self.current_block == "shocks":
        #     self.steady_states[name] = value
    
    def setvar_stmt(self, tree):
        name = str(tree.children[0].children[0].children[0])
        formula = (tree.children[1])
        value = self.visit(formula)
        self.covariances[(name, name)] = value
        return tree
    
    # Function calls
    def call(self, tree):
        """Handle function calls: func_name(arg)"""
        func_name = str(tree.children[0])
        args = [self.visit(c) for c in tree.children[1:]]
        
        if func_name in self.function_table:
            return self.function_table[func_name](*args)
        else:
            raise ValueError(f"Undefined function: {func_name}")
    
    def lequation(self, tree):
        self.equations.append(tree.children[1])
        # val =  self.visit(tree.children[1])
        # return val
    
    def equality(self, tree):

        # maybe not exactly conform
        a = self.visit(tree.children[0])
        b = self.visit(tree.children[1])
        return  b-a

from dyno.dynsym.latex import LatexTransformer

def parse_modfile(txt):

    trans = ModFileTransformer()
    parser = Lark(
        modfile_grammar,
        propagate_positions=True,
        parser="lalr",
        transformer=trans,
        cache=True,
        # strict=True,
    )
    tree = parser.parse(txt)


    fe = InterpretModfile(steady_state=True)
    fe.visit(tree)
    # for eq in tree.children:
    #     # if eq.data == "model_block":
    #     #     continue
    #     fe.visit(eq)

    return tree, trans, fe

tree, trans, fe = parse_modfile(modtxt)

fe.steady_state = True


import time
t1 = time.time()
tree, trans, fe = parse_modfile(modtxt)
res = [fe.visit(eq) for eq in fe.equations]
t2 = time.time()
print("Full interpretation time: ", t2-t1)



from dyno.dynsym.latex import latex
leq = []
for eq in fe.equations:
    leq.append(latex(eq))

from IPython.display import display, Math

for eq in leq:
    display(Math(eq))





