from dyno.symbolic_model import DynoModel as Model
from dyno.modfile import DynareModel


model = AbstractModel("../jupyterlab-dyno/examples/ramst.dyno")
dmodel = DynareModel("examples/modfiles/ramst.mod")


from dyno.solver import deterministic_solve

sim = deterministic_solve(model)

v0 = model.deterministic_guess()

r1 = model.deterministic_residuals(v0)

r2 = dmodel.deterministic_residuals(v0)
