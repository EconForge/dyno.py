from dyno.dyno_model import DynoModel
from dyno.modfile import DynareModel
from dyno import examples_path
import time
import numpy as np

from dyno.experimental.higher import HigherOrderDerivatives


def get_dynare_derivatives(mod_path):
    print("Running Dynare preprocessor...")
    model = DynareModel(mod_path, deriv_order=3)
    res = model.compute_derivatives()

    n = len(model.symbols["endogenous"])
    m = len(model.symbols["exogenous"])
    indices = {}
    for i, el in enumerate(model.symbolic.symbol_info):
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

    nres = [res[0]]
    for r in res[1:]:
        d = {}
        for el in r:
            ind, v = el
            # Sort the variable indices to match canonical format (skip equation index)
            var_indices = sorted([indices[e] for e in ind[1:]])
            ind2 = tuple([ind[0]] + var_indices)
            d[ind2] = d.get(ind2, 0.0) + v
        nres.append(d)
        
    return nres


def get_sympy_derivatives(mod_path):
    print("Running Sympy experimental API...")
    model = DynoModel(mod_path)
    ho = HigherOrderDerivatives(model)
    sym_derivatives = ho.compute_sparse_derivatives(max_order=3)
    num_derivatives = ho.evaluate_at_steady_state(sym_derivatives)
    return num_derivatives


def compare_dictionaries(name, dynare_dict, sympy_dict, tol=1e-8):
    print(f"\n--- Comparing {name} ---")
    dynare_keys = set(dynare_dict.keys())
    sympy_keys = set(sympy_dict.keys())

    all_keys = dynare_keys.union(sympy_keys)
    max_err = 0.0
    diff_count = 0
    missing_in_dynare = 0
    missing_in_sympy = 0

    for k in all_keys:
        val_d = dynare_dict.get(k, 0.0)
        # Note: Sympy derivatives in dev_higher_2 were sometimes negatively signed compared to Dynare?
        # Let's check the absolute values or just direct comparison first.
        # Wait, the user script `dev_higher_2.py` had:
        # ediff = {k: -v.subs(values_dict) for k, v in diff.items()}
        # Notice the negative sign! I will apply it here.
        val_s = -sympy_dict.get(k, 0.0) if k in sympy_dict else 0.0
        
        if abs(val_d) < tol and abs(val_s) < tol:
            continue
            
        if k not in dynare_dict:
            missing_in_dynare += 1
        if k not in sympy_dict:
            missing_in_sympy += 1
            
        err = abs(val_d - val_s)
        if err > tol:
            diff_count += 1
            max_err = max(max_err, err)
            if diff_count <= 5:
                print(f"Mismatch at {k}: Dynare = {val_d}, Sympy = {val_s}, diff = {err}")

    print(f"Total entries (Dynare > tol): {len([k for k, v in dynare_dict.items() if abs(v) > tol])}")
    print(f"Total entries (Sympy > tol): {len([k for k, v in sympy_dict.items() if abs(v) > tol])}")
    if diff_count == 0:
        print("✅ Match! No significant differences found.")
    else:
        print(f"❌ Found {diff_count} differences (tol={tol}). Max error: {max_err}")
        if missing_in_dynare: print(f"Missing in Dynare: {missing_in_dynare}")
        if missing_in_sympy: print(f"Missing in Sympy: {missing_in_sympy}")


if __name__ == "__main__":
    mod_path = examples_path("modfiles", "RBC.mod")
    
    t1 = time.time()
    dynare_res = get_dynare_derivatives(mod_path)
    t2 = time.time()
    print(f"Dynare computation time: {t2 - t1:.4f}s")
    
    t3 = time.time()
    sympy_res = get_sympy_derivatives(mod_path)
    t4 = time.time()
    print(f"Sympy computation time: {t4 - t3:.4f}s")

    # dynare_res[0] is residuals
    # dynare_res[1] is Jacobian
    # dynare_res[2] is Hessian
    # dynare_res[3] is 3rd derivatives
    
    # sympy_res[0] is Jacobian
    # sympy_res[1] is Hessian
    # sympy_res[2] is 3rd derivatives
    
    compare_dictionaries("Jacobian", dynare_res[1], sympy_res[0])
    if len(dynare_res) > 2 and len(sympy_res) > 1:
        compare_dictionaries("Hessian", dynare_res[2], sympy_res[1])
    if len(dynare_res) > 3 and len(sympy_res) > 2:
        compare_dictionaries("3rd Derivatives", dynare_res[3], sympy_res[2])
