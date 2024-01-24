import numpy as np

def jacobian(func,initial,delta=1e-3):
  
    f = func
    f0 = f(initial)
    nrow = len(f0)
    ncol = len(initial)
    output = np.zeros(nrow*ncol)
    output = output.reshape(nrow,ncol)
#   for i in range(nrow):
#     for j in range(ncol):
#       ej = np.zeros(ncol)
#       ej[j] = 1
#     #   dij = (f(initial+ delta * ej)[i] - f(initial- delta * ej)[i])/(2*delta)
#       dij = (f(initial+ delta * ej)[i] - f0[i])/(delta)
#       output[i,j] = dij
    for j in range(ncol):
        ej = np.zeros(ncol)
        ej[j] = 1
    #   dj = (f(initial+ delta * ej) - f(initial- delta * ej))/(2*delta)
        dj = (f(initial+ delta * ej) - f0)/(delta)
        output[:,j] = dj

    return output