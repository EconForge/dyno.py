# from dyno.gui.dynare import dyno_gui

# # Page = dyno_gui("examples/neo.yaml", parchoice={'beta':[0.99, 0.9, 0.99]})
# Page = dyno_gui("examples/modfiles/model_KR2000_STAT.mod", parchoice={"beta": [0.99, 0.9, 0.99]})


from dyno.modfile import Modfile

model = Modfile("examples/modfiles/model_KR2000_STAT.mod")

dr = model.solve()

import numpy as np

unconditional_moments(dr.X, dr.Y, dr.Σ)
