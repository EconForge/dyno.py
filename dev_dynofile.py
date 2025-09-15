from dyno.dynofile import *
from dyno.modfile import *

from rich import print

import time
model1 = DynoModel("RBC.dyno")
model2 = DynareModel("examples/modfiles/RBC.mod")

d1 = (model1.get_calibration())
d2 = (model2.calibration)

d = ( {k: (d1[k], d2[k], d1[k]-d2[k]) for k in d1.keys() | d2.keys() if k in d1 and k in d2 } )
res1 = model1.compute()
res2 = model2.compute()

# for u,v in zip(res1, res2):
#     print(u)
#     print(v)
#     print(u-v)
#     print("----")

dr1 = model1.solve()
dr2 = model2.solve()
vars1 = model1.symbols['endogenous']
vars2 = model2.symbols['endogenous']
import numpy as np
XX = np.concatenate([dr1.X[:,vars1.index(sym)][:,None] for sym in vars2], axis=1)
XX = np.concatenate([XX[vars1.index(sym),:][None,:] for sym in vars2], axis=0)


vare1 = model1.symbols['exogenous']
vare2 = model2.symbols['exogenous']
import numpy as np
YY = np.concatenate([dr1.Y[:,vare1.index(sym)][:,None] for sym in vare2], axis=1)
YY = np.concatenate([YY[vars1.index(sym),:][None,:] for sym in vars2], axis=0)

print(XX - dr2.X)
print(YY - dr2.Y)

# It works !


print(model1.symbols['endogenous'])
print(model2.symbols['endogenous'])



exit(0)



print("HI")
from dyno.dynofile import *

import time
model = DynoModel("RBC.dyno")
dr = model.solve()

print(model.name)
print(model.equations)
print(model.symbols)
print(model.exogenous)

t1 = time.time()
model = DynoModel("RBC.dyno")
dr0 = model.solve()
t2 = time.time()

print(f"Import  + solution time: {t2-t1} seconds")

print(dr)

### TEST MODFILE PREPROCESSOR


print("HI")
from dyno.modfile import *

import time
model = DynareModel("examples/modfiles/RBC.mod")
t1 = time.time()
model = DynareModel("examples/modfiles/RBC.mod")
t2 = time.time()

print(model.name)
print(model.equations)
print(model.symbols)
print(model.exogenous)

print(f"Import time: {t2-t1} seconds")

dr1 = model.solve()


from rich import print
print(dr0.X)
print(dr1.X)

print(dr1.X - dr0.X)

# ### TEST MODFILE PARSER

# from dyno.modfile_lark import *

# import time
# model = DynareModel("examples/modfiles/RBC.mod")
# t1 = time.time()
# model = DynareModel("examples/modfiles/RBC.mod")
# t2 = time.time()

# print(model.name)
# print(model.equations)
# print(model.symbols)
# print(model.exogenous)

# dr2 = model.solve()

# print(dr2.X - dr1.X)

# print(f"Import time: {t2-t1} seconds")