import os
import time

dir_path = os.path.dirname(os.path.realpath(__file__))


f_r = "examples/modfiles/RBC.mod"
f_o = "examples/RBC.dyno"

from dyno.symbolic_model import SymbolicModel

t1 = time.time()
model = SymbolicModel(f_o)
t2 = time.time()
dr1 = model.solve()
print("Model parsing time: ", t2 - t1)

from dyno.symbolic_model import SymbolicModel

t1 = time.time()
model = SymbolicModel(f_r)
t2 = time.time()
dr2 = model.solve()
t3 = time.time()
print("Model parsing time: ", t2 - t1)
# print("Model solving time: ", t3 - t2)

from dyno.modfile import DynareModel as DynareModelPP

t1 = time.time()
model = DynareModelPP(f_r)
t2 = time.time()
dr3 = model.solve()
t3 = time.time()
print("Model parsing time: ", t2 - t1)
# print("Model solving time: ", t3 - t2)


from IPython.display import Math, display

from dyno.dynsym.latex import latex

for eq in model.evaluator.equations:
    display(Math(latex(eq)))
