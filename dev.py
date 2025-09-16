from dyno.dynofile import DynoModel

from rich import print, inspect

model = DynoModel("examples/ramst.dyno")

# from dyno.modfile import DynareModel
# from dyno.modfile_lark import DynareModel
# model = DynareModel("examples/modfiles/ramst.mod")

inspect(model)

print(model.variables)


def deterministic_solve(model):

    T = model.calibration['T']
    T = 50
    import numpy as np

    y,e = model.steady_state()

    # initial guess
    v0 = np.concatenate([y,e])[None,:].repeat(T+1,axis=0)
    v1 = v0.copy()

    # works if the is one and exactly one exogenous variable?
    # does it?
    for key,value in model.evaluator.values.items():
        i = model.variables.index(key)
        for a,b in value.items():
            v1[a,i] = b

    exo = v1[:,-1].copy()

    def residuals(v):

        v_f = np.concatenate([v[1:,:], v[-1,:][None,:]], axis=0)
        v_b = np.concatenate([v[0,:][None,:], v[:-1,:]], axis=0)

        context = {}
        for i,name in enumerate(model.variables):
            context[name] = { -1: v_b[:,i], 0: v[:,i], 1: v_f[:,i] }

        E = (model.evaluator)
        E.variables.update(context)

        results = [E.visit(eq) for eq in E.equations]

        results.append((v[:,-1] - exo))

        return np.column_stack(results)

    import scipy.optimize

    from dyno.misc import jacobian
    J = jacobian(lambda u: residuals(u.reshape(v0.shape)).ravel(), v0.flatten())

    fobj = lambda u: residuals(u.reshape(v0.shape)).ravel()
    u0 = v0.flatten()
    res = scipy.optimize.fsolve(fobj, u0)

    w0 = res.reshape(v0.shape)

    df = pandas.DataFrame({e:w0[:,i] for i,e in enumerate(model.variables)})

    return df

import pandas

import time

t1 = time.time()

df = deterministic_solve(model)
t2 = time.time()

print(f"Solved in {t2-t1:.2f} seconds")

# plt.plot(w0[:,0])
# plt.plot(w0[:,1])
# plt.plot(w0[:,2])


T = model.calibration['T']
T = 50
import numpy as np

y,e = model.steady_state()

# initial guess
v0 = np.concatenate([y,e])[None,:].repeat(T+1,axis=0)
v1 = v0.copy()

# works if the is one and exactly one exogenous variable?
# does it?
for key,value in model.evaluator.values.items():
    i = model.variables.index(key)
    for a,b in value.items():
        v1[a,i] = b

exo = v1[:,-1].copy()

from dyno.dynsym.autodiff import DNumber


def residuals(v):

    v_f = np.concatenate([v[1:,:], v[-1,:][None,:]], axis=0)
    v_b = np.concatenate([v[0,:][None,:], v[:-1,:]], axis=0)

    context = {}
    for i,name in enumerate(model.variables):
        context[name] = {
            -1: v_b[:,i],
             0: v[:,i],
             1: v_f[:,i],
        }

    E = (model.evaluator)
    E.variables.update(context)

    results = [E.visit(eq) for eq in E.equations]

    results.append((v[:,-1] - exo))

    res = np.column_stack(results)
    res[0,:] = v[0,:] - y # slightly inconsistent

    return res

def residuals_with_grad(u):

    v = u.reshape(v0.shape)

    v_f = np.concatenate([v[1:,:], v[-1,:][None,:]], axis=0)
    v_b = np.concatenate([v[0,:][None,:], v[:-1,:]], axis=0)

    context = {}
    for i,name in enumerate(model.variables):
        context[name] = {
            -1: DNumber(v_b[:,i], {(name,-1): 1.0}),
             0: DNumber(v[:,i], {(name,0): 1.0}),
             1: DNumber(v_f[:,i],  {(name,1): 1.0})
        }

    E = (model.evaluator)
    E.variables.update(context)

    results = [E.visit(eq) for eq in E.equations]
    
    r = np.column_stack(
        [e.value for e in results] + [v[:,-1] - exo]
    )
    
    N = v.shape[0]

    p = len(model.variables)
    q = len(model.equations)

    dynvars = [(s,-1) for s in model.variables]  + [(s,0) for s in model.variables] + [(s,1) for s in model.variables]
    
    D = np.zeros( (N, q, p, 3 ))  # would be easier with 4d struct

    for i_q in range(q):
        
        for k,deriv in results[i_q].derivatives.items():
            s,t = k # symbol, time
            i_var = model.variables.index(s)
            D[:, i_q, i_var, t+1] = deriv

    # add exogenous equations
    DD = np.zeros( (N, p, p, 3))
    DD[:,:q,:,:] = D
    DD[:,2,2,1] = 1.0

    J = np.zeros((N*p, N*p))
    for n in range(N):
        if n==0:
            # J[p*n:p*(n+1),p*n:p*(n+1)] = DD[n,:,:,0] + DD[n,:,:,1]
            # J[p*n:p*(n+1),p*(n+1):p*(n+2)] = DD[n,:,:,2]
            J[p*n:p*(n+1),p*n:p*(n+1)] = np.eye(p,p)
        elif n==N-1:
            J[p*n:p*(n+1),p*(n-1):p*(n)] = DD[n,:,:,0]
            J[p*n:p*(n+1),p*n:p*(n+1)] = DD[n,:,:,1] + DD[n,:,:,2]
        else:
            J[p*n:p*(n+1),p*(n-1):p*(n)] = DD[n,:,:,0]
            J[p*n:p*(n+1),p*n:p*(n+1)] = DD[n,:,:,1]
            J[p*n:p*(n+1),p*(n+1):p*(n+2)] = DD[n,:,:,2]


    return r.ravel(), J


u0 = v0.ravel()


t1 = time.time()
rr, JJ = residuals_with_grad(u0);
t2 = time.time()
print(t2-t1)



residuals(v0);

import scipy.optimize

from dyno.misc import jacobian

t1 = time.time()
J = jacobian(lambda u: residuals(u.reshape(v0.shape)).ravel(), v0.flatten())
t2 = time.time()
print(t2-t1)

fobj = lambda u: residuals(u.reshape(v0.shape)).ravel()


u0 = v0.flatten()

t1 = time.time()
res = scipy.optimize.root(fobj, u0)
t2 = time.time()
print(f"Numerical Gradient: {t2-t1}")


t1 = time.time()
res2 = scipy.optimize.root(
    residuals_with_grad,
    u0,
    jac=True
)
t2 = time.time()
print(f"Exact Gradient: {t2-t1}")





w0 = res.reshape(v0.shape)

df = pandas.DataFrame({e:w0[:,i] for i,e in enumerate(model.variables)})
