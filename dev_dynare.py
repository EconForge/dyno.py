import os
import time
dir_path = os.path.dirname(os.path.realpath(__file__))

modfile_grammar = open(f"src/dyno/dynsym/grammars/modfile_grammar.lark").read()
modtxt = open(f"{dir_path}/examples/modfiles/example2.mod").read()
from lark import Tree, Token, Lark
from lark.visitors import Transformer

class Sanitizer(Transformer):

    def __init__(self):
        self.variables = []

    def var_statement(self, tree):
        for child in tree.children:
            s = children[0]
            print(s)



parser = Lark(
    modfile_grammar,
    propagate_positions=True,
    # parser="lalr",
    # strict=True,

)

tree = parser.parse(modtxt)


print(tree.pretty())
# t1 = time.time()
# try:
#     parser.parse(modtxt)
# except Exception as e:
#     print(e.get_context(modtxt))
#     raise e
# t2 = time.time()

# print(f"Parsing time: {t2-t1:.3f} seconds")


# t1 = time.time()
# parser.parse(modtxt)
# t2 = time.time()

# print(f"Parsing time: {t2-t1:.3f} seconds")