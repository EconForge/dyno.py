import dyno

from matplotlib import pyplot as plt
from dyno.symbolic_model import SymbolicModel
import pandas as pd

model = SymbolicModel("ICAI/partial_deterministic.dyno")
model2  = SymbolicModel("ICAI/partial_shock.dyno")
dr2 = model2.solve()
sim2 = dr2.irfs(type='level')['e_d'] # level seems to be ignored


y,e = model.__steady_state_vectors__

v0 = model.deterministic_guess()

pd.DataFrame(v0, columns=model.symbols['variables'])

from dyno.solver import deterministic_solve
from dyno.solver import solve as linsolve
import time


r0,A,B,C,D = model.jacobians
X, evs = linsolve(A,B,C)
r, J = model.deterministic_residuals_with_jacobian(v0.ravel(), sparsify=True)
rr = r.reshape(v0.shape)





import numpy as np

np.isnan(r).sum()
np.isnan(J.todense()).sum()

from scipy.sparse.linalg import spsolve

sims = [sim2]
for t in [150,200,250]:

    m = model.recalibrate(T=t)

    t1 = time.time()
    sim = deterministic_solve(m)
    t2 = time.time()
    print("Elapsed time: ", t2 - t1)
    sims.append(sim)

sims.reverse()





plt.figure()

for i,sim in enumerate(sims):
    plt.subplot(221)
    plt.plot(sim.index, sim['W_τ'], label=i)
    plt.ylim(2.4, 2.5)
    plt.legend()
    plt.xlim(0,50)
    plt.subplot(222)
    plt.plot(sim.index, sim['c_τ'])
    plt.xlim(0,50)
    plt.subplot(223)
    plt.plot(sim.index, sim['d'])
    try:
        plt.subplot(224)
        plt.plot(sim.index, sim['e_d'])
        plt.ylim(-0.001, 0.01)
    except:
        pass


plt.xlim(0,20)
plt.tight_layout()

plt.legend()







model.jacobians[0]


dr = model.solve()
sim = dr.irfs(type='level')['e_d']
# plt.plot(sim.index, sim['W_τ'], label="baseline")

plt.plot(sim.index, sim['b_b'], label="baseline",marker='.',linestyle='-')
plt.plot(sim.index, sim['b_τ'], label="baseline",marker='.',linestyle='-')
plt.xlim(0,10)

plt.plot(sim.index, sim['W_b'], label="baseline",marker='.',linestyle='-')
plt.plot(sim.index, sim['W_τ'], label="baseline",marker='.',linestyle='-')
plt.xlim(0,10)


m2 = model.recalibrate(η=1.51)
dr2 = m2.solve()
sim2 = dr2.irfs()['e_d']

# plt.plot(sim.index, (sim['W_τ']-sim['W_τ'][0])/sim['d'], label="baseline")
# plt.plot(sim2.index, (sim2['W_τ']-sim2['W_τ'][0])/sim2['d'], label="η=3.0")

plt.plot(sim2.index, sim2['W_τ'])


m2 = model.recalibrate(η=1.51)
dr2 = m2.solve()
sim2 = dr2.irfs()['e_d']

# plt.plot(sim.index, (sim['W_τ']-sim['W_τ'][0])/sim['d'], label="baseline")
# plt.plot(sim2.index, (sim2['W_τ']-sim2['W_τ'][0])/sim2['d'], label="η=3.0")

plt.plot(sim.index, (sim['W_b']-sim['W_b'][0])/sim['d'], label="baseline")
plt.plot(sim2.index, (sim2['W_b']-sim2['W_b'][0])/sim2['d'], label="η=3.0")





print(m2.data.context['constants']['D'])
print(m2.data.context['processes'][('e_d',)].Σ)


plt.plot(sim['e_d'].index, sim['e_d']['d'], label="baseline")
plt.plot(sim2['e_d'].index, sim2['e_d']['d'], label="D=0.05")
plt.legend()


###### 

import dyno

from matplotlib import pyplot as plt

from dyno.symbolic_model import SymbolicModel


model = SymbolicModel("partial.dyno")

dr = model.solve()

irfs = dr.irfs()

# sim = dyno.simulate(dr)

plt.figure()

plt.subplot(121)
plt.plot(irfs['e_d'].index, irfs['e_d']['d']*0, color='black', linestyle='--')
plt.plot(irfs['e_d'].index, irfs['e_d']['d'],label='Income')
plt.xlim(0, 10)
plt.subplot(122)
plt.plot(irfs['e_d'].index, irfs['e_d']['b_τ'],label='Debt')
plt.xlim(0, 10)
# plt.ylim(0, 0.02)



plt.plot(irfs['e_d'].index, irfs['e_d']['d'],label='Income')
plt.plot(irfs['e_d'].index, irfs['e_d']['b_τ'],label='Debt')
plt.xlim(0, 10)
# plt.ylim(0, 0.02)


plt.plot(irfs['e_d'].index, irfs['e_d']['b_τ']/irfs['e_d']['d'],label='MPS')
plt.xlim(0,10)
