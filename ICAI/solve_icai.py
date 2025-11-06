import dyno
from matplotlib import pyplot as plt
from dyno.symbolic_model import SymbolicModel

model = SymbolicModel("partial.dyno")

dr = model.solve()


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
