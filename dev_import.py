from dyno.symbolic_model import DynoModel
from dyno.modfile import DynareModel

from rich import inspect, print

model1 = DynoModel("examples/RBC.dyno")
model2 = DynoModel("examples/modfiles/RBC.mod")
model3 = DynareModel("examples/modfiles/RBC.mod")


models = [model1, model2, model3]

print("Filenames of imported models:")
print([model.filename for model in models])


print("Names of imported models:")
print([model.name for model in models])

print("Symbols of imported models:")
for m in models:
    print(m.filename)
    print(m.symbols)


print("Context of imported models:")
for m in models:
    print(m.filename)
    print(m.context)


print("Equations of imported models:")
for m in models:
    print(m.equations)


print("Equations of imported models:")
for m in models:
    print(m.equations)

for m in models:
    print(m.filename)
    print(m.steady_state)

for m in models:

    print(m.__steady_state_vectors__)

for m in models:
    print(m.filename)
    print(m.residuals)

for m in models:
    print(m.filename)
    print(m.jacobians)

for m in models:
    dr = print(m._repr_html_())


for m in models:
    dr = m.solve()
    print(dr)


import time


t1 = time.time()
for i in range(100):
    model = DynoModel("examples/modfiles/RBC.mod")
    dr = model.solve()
t2 = time.time()
print("Elsapsed time (dyno/mod)", t2 - t1)


# Importing modfile with preprocessor
from dyno.modfile import DynareModel
import time

t1 = time.time()
for i in range(100):
    model = DynareModel("examples/modfiles/RBC.mod")
    model.solve()
t2 = time.time()
print("Elsapsed time (preprocessos)", t2 - t1)

# Importing modfile with preprocessor
from dyno.modfile import DynareModel
import time

t1 = time.time()
for i in range(100):
    model = DynoModel("examples/RBC.dyno")
    model.solve()
t2 = time.time()
print("Elsapsed time (dyno/dyno):", t2 - t1)


# exit()
