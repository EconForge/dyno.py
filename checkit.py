filename = "examples/modfiles/example1.mod"

from dyno.modfile_lark import DynareModel

model = DynareModel(filename=filename)

dr = model.solve()

print(dr)