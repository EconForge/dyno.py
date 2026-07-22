from dyno.dyno_model import DynoModel
from dyno import examples_path
import time

model = DynoModel(examples_path("modfiles", "RBC.mod"))

from dyno.experimental.higher import (
    HigherOrderDerivatives,
    sparse_coo_to_dense
)

# Optional helper to print summaries (migrated to local or just simple print)
def print_sparse_summary(sparse_coo_dict, name="Sparse tensor"):
    """Print summary statistics for sparse COO dictionary."""
    total_entries = len(sparse_coo_dict)
    if total_entries == 0:
        print(f"{name}: No non-zero entries")
        return
    print(f"{name}: {total_entries} non-zero entries")
    print("First few entries:")
    for i, (indices, value) in enumerate(sparse_coo_dict.items()):
        if i >= 5:
            print("  ...")
            break
        print(f"  {indices}: {value}")


# 1. Initialize API
print("Building Sympy representation...")
ho = HigherOrderDerivatives(model)

# 2. Compute Symbolic Derivatives
print("Computing derivatives symbolically up to order 3...")
t1 = time.time()
sym_derivatives = ho.compute_sparse_derivatives(max_order=3)
t2 = time.time()
print("Symbolic derivatives time:", t2 - t1)

if len(sym_derivatives) >= 1:
    print_sparse_summary(sym_derivatives[0], "Jacobian")
if len(sym_derivatives) >= 2:
    print_sparse_summary(sym_derivatives[1], "Hessian")
if len(sym_derivatives) >= 3:
    print_sparse_summary(sym_derivatives[2], "3rd derivatives")

# 3. Evaluate numerically at steady state
print("\nEvaluating numerically at steady state...")
t1 = time.time()
num_derivatives = ho.evaluate_at_steady_state(sym_derivatives)
t2 = time.time()
print("Numerical evaluation time:", t2 - t1)

if len(num_derivatives) >= 1:
    print_sparse_summary(num_derivatives[0], "Numerical Jacobian")

print("\nImplementation complete!")
