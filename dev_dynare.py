import os
import time

dir_path = os.path.dirname(os.path.realpath(__file__))


f_o = "examples/RBC.dyno"
f_r = "examples/modfiles/RBC.mod"


from dyno.symbolic_model import SymbolicModel

model1 = SymbolicModel(f_o)
model2 = SymbolicModel(f_r)


import copy

t1 = time.time()
m1 = model1.recalibrate(beta=0.4)
t2 = time.time()
print("Elapsed: ", t2 - t1)



import copy
t1 = time.time()
mm1 = copy.deepcopy(model1)
mm1.solve()
t2 = time.time()
print("Elapsed: ", t2 - t1)


import copy
t1 = time.time()
mm2 = copy.deepcopy(model2)
dr = mm2.solve()
t2 = time.time()
print("Elapsed: ", t2 - t1)



from dyno.symbolic_model import SymbolicModel

t1 = time.time()
model = SymbolicModel(f_o)
t2 = time.time()
dr1 = model.solve()
t3 = time.time()
print("Model parsing time: ", t2 - t1)
print("Model solving time: ", t3 - t2)


from dyno.symbolic_model import SymbolicModel

t1 = time.time()
model = SymbolicModel(f_r)
t2 = time.time()
dr2 = model.solve()
t3 = time.time()
print("Model parsing time: ", t2 - t1)
print("Model solving time: ", t3 - t2)






from dyno.modfile import DynareModel as DynareModelPP

t1 = time.time()
model = DynareModelPP(f_r)
t2 = time.time()
dr3 = model.solve()
t3 = time.time()
print("Model parsing time: ", t2 - t1)
print("Model solving time: ", t3 - t2)


from IPython.display import Math, display

from dyno.dynsym.latex import latex

for eq in model.evaluator.equations:
    display(Math(latex(eq)))
