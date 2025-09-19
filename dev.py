from dyno.dynofile import DynoModel

model = DynoModel('../jupyterlab-dyno/examples/ramst.dyno')

from dyno.solver import deterministic_solve

deterministic_solve(model)