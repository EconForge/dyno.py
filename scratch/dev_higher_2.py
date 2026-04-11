from dyno.symbolic_model import DynoModel

model = DynoModel("examples/modfiles/RBC.mod")

import sympy as sp


from dyno.dynsym.grammar import (
    stringify_symbol,
    stringify_variable,
    stringify_value,
    stringify_constant,
)

model.symbolic.equations

v_context = model.symbolic.context["variables"]
for v in model.symbols["variables"]:
    v_context[v] = {
        -1: sp.Symbol(stringify_variable((v, "t", -1))),
        0: sp.Symbol(stringify_variable((v, "t", 0))),
        1: sp.Symbol(stringify_variable((v, "t", 1))),
    }
import copy

evaluator = model.symbolic.evaluator
evaluator.function_table["log"] = sp.log
evaluator.function_table["exp"] = sp.exp
evaluator.function_table["sqrt"] = sp.sqrt
evaluator.function_table["min"] = sp.Min
evaluator.function_table["max"] = sp.Max
evaluator.function_table["abs"] = sp.Abs
evaluator.function_table["pow"] = sp.Pow

residuals = [model.symbolic.evaluator.visit(eq) for eq in model.symbolic.equations]

symbols = (
    [v_context[v][+1] for v in model.symbols["endogenous"]]
    + [v_context[v][0] for v in model.symbols["endogenous"]]
    + [v_context[v][-1] for v in model.symbols["endogenous"]]
    + [v_context[v][0] for v in model.symbols["exogenous"]]
)
values = [
    model.symbolic.context["steady_states"][e] for e in model.symbols["endogenous"]
] * 3 + [model.symbolic.context["steady_states"][e] for e in model.symbols["exogenous"]]
values_dict = dict(zip(symbols, values))


import time

t1 = time.time()
jacobian = sp.Matrix(residuals).jacobian(symbols)
diff = compute_sparse_jacobian(residuals, symbols)
ediff = {k: -v.subs(values_dict) for k, v in diff.items()}
diff3 = compute_sparse_third_derivatives(residuals, symbols)
ediff3 = {k: -v.subs(values_dict) for k, v in diff3.items()}
t2 = time.time()
print("Symbolic jacobian time:", t2 - t1)

# comparison
A = sorted([e[1] for e in tt[-1].data])
B = sorted([e for e in ediff3.values()])

# Sparse derivatives computation functions
import itertools
from collections import defaultdict


def canonical_index(indices):
    """Convert multi-index to canonical form by sorting the derivative indices.
    For derivatives, the order of differentiation doesn't matter due to symmetry.

    Args:
        indices: tuple of (equation_idx, var1_idx, var2_idx, ...)
                where equation_idx stays fixed, but var indices are sorted

    Returns:
        tuple: canonical form with sorted variable indices
    """
    if len(indices) <= 1:
        return indices
    # Keep first index (equation), sort the rest (variables)
    return (indices[0],) + tuple(sorted(indices[1:]))


def compute_sparse_jacobian(residuals, symbols):
    """Compute Jacobian in sparse COO format.

    Args:
        residuals: list of sympy expressions
        symbols: list of sympy symbols

    Returns:
        dict: {(i, j): derivative_value} where derivative_value is sympy expression
    """
    jacobian_coo = {}

    for i, residual in enumerate(residuals):
        for j, symbol in enumerate(symbols):
            deriv = sp.diff(residual, symbol)
            if deriv != 0:  # Only store non-zero entries
                jacobian_coo[(i, j)] = deriv

    return jacobian_coo


def compute_sparse_hessian(residuals, symbols):
    """Compute Hessian (2nd derivatives) in sparse COO format avoiding symmetric terms.

    Args:
        residuals: list of sympy expressions
        symbols: list of sympy symbols

    Returns:
        dict: {(i, j, k): derivative_value} with canonical indices
    """
    hessian_coo = {}

    for i, residual in enumerate(residuals):
        # Use combinations_with_replacement to avoid symmetric duplicates
        for j, k in itertools.combinations_with_replacement(range(len(symbols)), 2):
            deriv = sp.diff(residual, symbols[j], symbols[k])
            if deriv != 0:
                canonical_idx = canonical_index((i, j, k))
                hessian_coo[canonical_idx] = deriv

    return hessian_coo


def compute_sparse_third_derivatives(residuals, symbols):
    """Compute 3rd derivatives in sparse COO format avoiding symmetric terms.

    Args:
        residuals: list of sympy expressions
        symbols: list of sympy symbols

    Returns:
        dict: {(i, j, k, l): derivative_value} with canonical indices
    """
    third_deriv_coo = {}

    for i, residual in enumerate(residuals):
        # Use combinations_with_replacement to avoid symmetric duplicates
        for j, k, l in itertools.combinations_with_replacement(range(len(symbols)), 3):
            deriv = sp.diff(residual, symbols[j], symbols[k], symbols[l])
            if deriv != 0:
                canonical_idx = canonical_index((i, j, k, l))
                third_deriv_coo[canonical_idx] = deriv

    return third_deriv_coo


def compute_all_derivatives(residuals, symbols, max_order=3):
    """Compute derivatives up to specified order in sparse COO format.

    Args:
        residuals: list of sympy expressions
        symbols: list of sympy symbols
        max_order: maximum derivative order (1=Jacobian, 2=Hessian, 3=third derivatives)

    Returns:
        list: [jacobian_coo, hessian_coo, third_deriv_coo] up to max_order
    """
    derivatives = []

    if max_order >= 1:
        print("Computing Jacobian...")
        jac = compute_sparse_jacobian(residuals, symbols)
        derivatives.append(jac)
        print(f"Jacobian: {len(jac)} non-zero entries")

    if max_order >= 2:
        print("Computing Hessian...")
        hess = compute_sparse_hessian(residuals, symbols)
        derivatives.append(hess)
        print(f"Hessian: {len(hess)} non-zero entries")

    if max_order >= 3:
        print("Computing 3rd derivatives...")
        third = compute_sparse_third_derivatives(residuals, symbols)
        derivatives.append(third)
        print(f"3rd derivatives: {len(third)} non-zero entries")

    return derivatives


# Additional utility functions


def expand_symmetric_indices(sparse_coo_dict):
    """Expand canonical sparse COO dict to include all symmetric permutations.

    Args:
        sparse_coo_dict: dict with canonical indices as keys

    Returns:
        dict: expanded dict with all symmetric permutations
    """
    expanded = {}

    for indices, value in sparse_coo_dict.items():
        if len(indices) <= 2:  # Jacobian case - no symmetry to expand
            expanded[indices] = value
        else:
            # Generate all unique permutations of variable indices (keeping equation index fixed)
            eq_idx = indices[0]
            var_indices = indices[1:]

            # Generate all permutations and add them
            for perm in itertools.permutations(var_indices):
                full_idx = (eq_idx,) + perm
                expanded[full_idx] = value

    return expanded


def sparse_coo_to_dense(sparse_coo_dict, shape):
    """Convert sparse COO format to dense array (for small problems only).

    Args:
        sparse_coo_dict: dict with indices as keys, values as entries
        shape: tuple of dimensions

    Returns:
        numpy array or nested lists with the dense representation
    """
    import numpy as np

    if len(shape) == 2:  # Jacobian
        dense = np.zeros(shape, dtype=object)
        for (i, j), value in sparse_coo_dict.items():
            dense[i, j] = value
    elif len(shape) == 3:  # Hessian
        dense = np.zeros(shape, dtype=object)
        # Expand symmetric terms
        expanded = expand_symmetric_indices(sparse_coo_dict)
        for (i, j, k), value in expanded.items():
            dense[i, j, k] = value
    elif len(shape) == 4:  # 3rd derivatives
        dense = np.zeros(shape, dtype=object)
        # Expand symmetric terms
        expanded = expand_symmetric_indices(sparse_coo_dict)
        for (i, j, k, l), value in expanded.items():
            dense[i, j, k, l] = value
    else:
        raise ValueError(f"Unsupported shape dimension: {len(shape)}")

    return dense


def print_sparse_summary(sparse_coo_dict, name="Sparse tensor"):
    """Print summary statistics for sparse COO dictionary."""
    total_entries = len(sparse_coo_dict)
    if total_entries == 0:
        print(f"{name}: No non-zero entries")
        return

    # Analyze sparsity pattern
    print(f"{name}: {total_entries} non-zero entries")

    # Show first few entries as examples
    print("First few entries:")
    for i, (indices, value) in enumerate(sparse_coo_dict.items()):
        if i >= 5:  # Show only first 5
            print("  ...")
            break
        print(f"  {indices}: {value}")


# # Test the implementation
# print(f"Computing derivatives for {len(residuals)} equations and {len(symbols)} variables...")
# derivatives = compute_all_derivatives(residuals, symbols, max_order=3)

# # Print summaries
# if len(derivatives) >= 1:
#     print_sparse_summary(derivatives[0], "Jacobian")
# if len(derivatives) >= 2:
#     print_sparse_summary(derivatives[1], "Hessian")
# if len(derivatives) >= 3:
#     print_sparse_summary(derivatives[2], "3rd derivatives")

# # Example: Access specific derivatives
# print("\nExample usage:")
# jac = derivatives[0] if len(derivatives) > 0 else {}
# print(f"Jacobian d(residual_0)/d(symbol_0): {jac.get((0, 0), 'Not found')}")

# if len(derivatives) > 1:
#     hess = derivatives[1]
#     print(f"Hessian d²(residual_0)/d(symbol_0)d(symbol_1): {hess.get((0, 0, 1), 'Not found')}")

# if len(derivatives) > 2:
#     third = derivatives[2]
#     print(f"3rd deriv d³(residual_0)/d(symbol_0)d(symbol_1)d(symbol_2): {third.get((0, 0, 1, 2), 'Not found')}")

# # Verification: test symmetry elimination
# print("\nVerifying symmetry elimination:")
# if len(derivatives) > 1:
#     hess = derivatives[1]
#     # Check that we only have canonical indices
#     for indices in hess.keys():
#         if len(indices) > 2:
#             var_indices = indices[1:]
#             canonical_var_indices = tuple(sorted(var_indices))
#             if var_indices != canonical_var_indices:
#                 print(f"ERROR: Found non-canonical index {indices}")
#             else:
#                 print(f"✓ Canonical index: {indices}")
#                 break

#     # Show how many entries we saved by avoiding symmetric terms
#     n_vars = len(symbols)
#     n_eqs = len(residuals)
#     total_possible_hess = n_eqs * n_vars * n_vars
#     canonical_possible_hess = n_eqs * n_vars * (n_vars + 1) // 2
#     print(f"Hessian: {total_possible_hess} total possible vs {canonical_possible_hess} canonical possible")
#     print(f"Memory savings: {(total_possible_hess - canonical_possible_hess) / total_possible_hess * 100:.1f}%")

# if len(derivatives) > 2:
#     third = derivatives[2]
#     n_vars = len(symbols)
#     n_eqs = len(residuals)
#     total_possible_third = n_eqs * n_vars ** 3
#     canonical_possible_third = n_eqs * (n_vars * (n_vars + 1) * (n_vars + 2)) // 6
#     print(f"3rd derivatives: {total_possible_third} total possible vs {canonical_possible_third} canonical possible")
#     print(f"Memory savings: {(total_possible_third - canonical_possible_third) / total_possible_third * 100:.1f}%")


# Additional utility: numerical evaluation
def evaluate_sparse_derivatives(sparse_derivatives_list, symbols, values_dict):
    """Evaluate sparse derivatives numerically at given point.

    Args:
        sparse_derivatives_list: list of sparse COO dicts (from compute_all_derivatives)
        symbols: list of sympy symbols
        values_dict: dict mapping symbols to numerical values

    Returns:
        list: [jac_values, hess_values, third_values] with numerical results
    """
    numerical_results = []

    for order, sparse_dict in enumerate(sparse_derivatives_list):
        numerical_dict = {}
        for indices, expr in sparse_dict.items():
            try:
                numerical_value = float(expr.subs(values_dict))
                numerical_dict[indices] = numerical_value
            except (TypeError, ValueError):
                # Keep symbolic if cannot evaluate numerically
                numerical_dict[indices] = expr
        numerical_results.append(numerical_dict)
        print(f"Order {order+1}: {len(numerical_dict)} entries evaluated")

    return numerical_results


print("\nImplementation complete!")
print("Functions available:")
print("- compute_sparse_jacobian(residuals, symbols)")
print("- compute_sparse_hessian(residuals, symbols)")
print("- compute_sparse_third_derivatives(residuals, symbols)")
print("- compute_all_derivatives(residuals, symbols, max_order=3)")
print("- canonical_index(indices)")
print("- expand_symmetric_indices(sparse_coo_dict)")
print("- sparse_coo_to_dense(sparse_coo_dict, shape)")
print("- evaluate_sparse_derivatives(sparse_derivatives_list, symbols, values_dict)")
