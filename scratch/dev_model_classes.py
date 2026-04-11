from dyno.larkfiles import DynoFile, LModFile
from rich import inspect, print

# TODO:  test that the two model classes give the same results

# f = DynoFile(open("examples/RBC.dyno").read())

# f = LModFile(open("examples/modfiles/RBC.mod").read())

fname_mod = "examples/modfiles/RBC.mod"
fname_dyno = "examples/RBC.dyno"
# f = LModFile(open("examples/RBC.dyno").read())
f = DynoFile(open(fname_dyno).read())
print(f.context)
txt1 = f.latex_equations()
f = LModFile(open(fname_mod).read())

print(f.context)
txt2 = f.latex_equations()

# the two models have exactly the same equations
assert txt1 == txt2
