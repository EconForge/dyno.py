from dyno.larkfiles import DynoFile, LModFile
from dyno import examples_path
from rich import inspect, print

# TODO:  test that the two model classes give the same results

# f = DynoFile(open(examples_path("RBC.dyno"), encoding="utf-8").read())

# f = LModFile(open(examples_path("modfiles", "RBC.mod"), encoding="utf-8").read())

fname_mod = examples_path("modfiles", "RBC.mod")
fname_dyno = examples_path("RBC.dyno")
# f = LModFile(open(examples_path("RBC.dyno"), encoding="utf-8").read())
f = DynoFile(open(fname_dyno, encoding="utf-8").read())
print(f.context)
txt1 = f.latex_equations()
f = LModFile(open(fname_mod, encoding="utf-8").read())

print(f.context)
txt2 = f.latex_equations()

# the two models have exactly the same equations
assert txt1 == txt2
