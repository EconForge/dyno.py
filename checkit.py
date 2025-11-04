filename = "examples/modfiles/example1.mod"

from dyno.dynsym.dynare import DynareModel

model = DynareModel(filename=filename)

dr = model.solve()

print(dr)
