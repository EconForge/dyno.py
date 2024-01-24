import numpy as np

def jacobian(func,initial,delta=1e-3):
  f = func
  nrow = len(f(initial))
  ncol = len(initial)
  output = np.zeros(nrow*ncol)
  output = output.reshape(nrow,ncol)
  for i in range(nrow):
    for j in range(ncol):
      ej = np.zeros(ncol)
      ej[j] = 1
      dij = (f(initial+ delta * ej)[i] - f(initial- delta * ej)[i])/(2*delta)
      output[i,j] = dij
  return output