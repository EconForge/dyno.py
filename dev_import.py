from dynsym import read_model
from rich import print, inspect

filename = "tests/rbc.dyno"

# sol = read_model("tests/rbc.dyno")
# # r, A, B, C, D = model.solve()
# print(sol)

txt = open(filename).read()
# print(txt)

from dynsym.analyze import FormulaEvaluator
from dynsym.grammar import parser

tree = parser.parse(txt, start="free_block")

from rich import print
print(tree.children[-1].pretty())

# print(tree.pretty())

fe = FormulaEvaluator(diff=False, steady_state=True)

res = fe.visit(tree)

print(fe.symbol_table)

# print(res)
# inspect(fe.symbols)

# # inspect(res)
def import_model(filename):
    txt = open(filename).read()
    tree = parser.parse(txt, start="free_block")
    fe = FormulaEvaluator(diff=False, steady_state=True)
    return fe.visit(tree), fe

from time import time

t1 = time()
res, an = import_model(filename)
t2 = time()

inspect(an)
print(f"Import time: {t2-t1:.4f} seconds")