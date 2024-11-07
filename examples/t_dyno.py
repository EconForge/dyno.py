from dyno.gui.dynare import dyno_gui

Page = dyno_gui("examples/modfiles/RBC.mod", parchoice={'beta':[0.99, 0.9, 0.99]})

# Page = dyno_gui("examples/neo.yaml", parchoice={'beta':[0.99, 0.9, 0.99]})
