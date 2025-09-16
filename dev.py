from dyno.dynofile import DynoModel

from rich import print, inspect

model = DynoModel("examples/ramst.dyno")

# from dyno.modfile import DynareModel
# from dyno.modfile_lark import DynareModel
# model = DynareModel("examples/modfiles/ramst.mod")

inspect(model)
# dr = model.solve()

# from rich import inspect
# inspect(dr)