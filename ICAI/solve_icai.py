import dyno
from matplotlib import pyplot as plt
from dyno.symbolic_model import DynoModel
import pandas as pd

model = DynoModel("partial.dyno")

v0 = model.deterministic_guess()

model.deterministic_residuals(v0.ravel())


df = pd.DataFrame(v0, columns=model.symbols['variables'])
df



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

from dyno.symbolic_model import DynoModel


model = DynoModel("partial.dyno")

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
