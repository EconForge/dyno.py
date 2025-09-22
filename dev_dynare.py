import os


import time
dir_path = os.path.dirname(os.path.realpath(__file__))


f_r = "examples/modfiles/RBC.mod"
f_o = "examples/RBC.dyno"

from dyno.dynofile import DynoModel

t1 = time.time()
model = DynoModel(f_o);
dr1 = model.solve();
t2 = time.time()
print("Model parsing time: ", t2-t1)

from dyno.modfile_lark import DynareModel

t1 = time.time()
model = DynareModel(f_r);
dr2 = model.solve();
t2 = time.time()
print("Model parsing time: ", t2-t1)




from dyno.modfile import DynareModel as DynareModelPP

t1 = time.time()
model = DynareModelPP(f_r);
dr3 = model.solve();
t2 = time.time()
print("Model parsing time: ", t2-t1)


from IPython.display import Math, display

from  dyno.dynsym.latex import latex
for eq in model.evaluator.equations:
    display(Math(latex(eq)))


