from dyno.modfile import DynareModel


def do_it():

    model = DynareModel("examples/modfiles/RBC.mod", deriv_order=3)

    res = model.compute_derivatives()
    return res


res = do_it()
import time

t1 = time.time()

do_it()
t2 = time.time()
print("Elapsed:", t2 - t1)

from rich import print

for r in res:
    print(r)

model = DynareModel("examples/modfiles/RBC.mod", deriv_order=3)

n = len(model.symbols["endogenous"])
m = len(model.symbols["exogenous"])
indices = {}
for i, el in enumerate(model.data.symbol_info):
    symtype = el[0].name
    if symtype == "endogenous":
        k, p = el[1], el[2]
        j = n - p * n + k
        indices[i] = j
    elif symtype == "exogenous":
        k, p = el[1], el[2]
        j = 3 * n + k
        indices[i] = j
    elif symtype == "parameter":
        pass
    else:
        raise Exception(f"Unknown symbol type {symtype}")

t1 = time.time()
nres = [res[0]]
for r in res[1:]:
    d = []
    for el in r:
        ind, v = el
        ind2 = [ind[0]] + [indices[e] for e in ind[1:]]
        d.append((ind2, v))
    nres.append(d)
t2 = time.time()

tt = [SymTensor(r, (n,) + (3 * n + m,) * i) for i, r in enumerate(nres)]
print("Reindexing time:", t2 - t1)


class SymTensor:
    "All dimensions are symmetric, except the first one"

    def __init__(self, data, shape):

        self.data = data
        self.shape = shape

    @property
    def ndims(self):
        return len(self.shape)
