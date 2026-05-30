import itertools
from typing import Dict, List, Tuple, Any
import sympy as sp

from dyno.dyno_model import DynoModel
from dyno.dynsym.grammar import stringify_variable
from dyno.dynsym.analyze import EquationsEvaluator, function_table_0


def canonical_index(indices: Tuple[int, ...]) -> Tuple[int, ...]:
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


def expand_symmetric_indices(sparse_coo_dict: Dict[Tuple[int, ...], Any]) -> Dict[Tuple[int, ...], Any]:
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
            for perm in set(itertools.permutations(var_indices)):
                full_idx = (eq_idx,) + perm
                expanded[full_idx] = value

    return expanded


def sparse_coo_to_dense(sparse_coo_dict: Dict[Tuple[int, ...], Any], shape: Tuple[int, ...]) -> Any:
    """Convert sparse COO format to dense array (for small problems only).

    Args:
        sparse_coo_dict: dict with indices as keys, values as entries
        shape: tuple of dimensions

    Returns:
        numpy array with the dense representation
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


class HigherOrderDerivatives:
    def __init__(self, model: DynoModel):
        self.model = model
        self.residuals, self.symbols, self.v_context = self._build_sympy_representation(model)

    def _build_sympy_representation(self, model: DynoModel):
        function_table = function_table_0.copy()
        
        v_context = {}
        for v in model.symbols["variables"]:
            v_context[v] = {
                -1: sp.Symbol(stringify_variable((v, "t", -1))),  # type: ignore
                0: sp.Symbol(stringify_variable((v, "t", 0))),    # type: ignore
                1: sp.Symbol(stringify_variable((v, "t", 1))),    # type: ignore
            }

        context = {
            "variables": v_context,
            "constants": model.symbolic.context.get("constants", {}),
        }

        # Create a localized evaluator that won't mutate the model's global state
        evaluator = EquationsEvaluator(context=context)
        evaluator.function_table.update({
            "log": sp.log,
            "exp": sp.exp,
            "sqrt": sp.sqrt,
            "min": sp.Min,
            "max": sp.Max,
            "abs": sp.Abs,
            "pow": sp.Pow,
        })

        residuals = [evaluator.visit(eq) for eq in model.symbolic.equations]

        symbols = (
            [v_context[v][+1] for v in model.symbols["endogenous"]]
            + [v_context[v][0] for v in model.symbols["endogenous"]]
            + [v_context[v][-1] for v in model.symbols["endogenous"]]
            + [v_context[v][0] for v in model.symbols["exogenous"]]
        )

        return residuals, symbols, v_context

    def compute_sparse_jacobian(self) -> Dict[Tuple[int, ...], sp.Expr]:
        jacobian_coo: Dict[Tuple[int, ...], sp.Expr] = {}
        for i, residual in enumerate(self.residuals):
            for j, symbol in enumerate(self.symbols):
                deriv = sp.diff(residual, symbol)
                if deriv != 0:
                    jacobian_coo[(i, j)] = deriv
        return jacobian_coo

    def compute_sparse_hessian(self) -> Dict[Tuple[int, ...], sp.Expr]:
        hessian_coo: Dict[Tuple[int, ...], sp.Expr] = {}
        for i, residual in enumerate(self.residuals):
            for j, k in itertools.combinations_with_replacement(range(len(self.symbols)), 2):
                deriv = sp.diff(residual, self.symbols[j], self.symbols[k])
                if deriv != 0:
                    canonical_idx = canonical_index((i, j, k))
                    hessian_coo[canonical_idx] = deriv
        return hessian_coo

    def compute_sparse_third_derivatives(self) -> Dict[Tuple[int, ...], sp.Expr]:
        third_deriv_coo: Dict[Tuple[int, ...], sp.Expr] = {}
        for i, residual in enumerate(self.residuals):
            for j, k, l in itertools.combinations_with_replacement(range(len(self.symbols)), 3):
                deriv = sp.diff(residual, self.symbols[j], self.symbols[k], self.symbols[l])
                if deriv != 0:
                    canonical_idx = canonical_index((i, j, k, l))
                    third_deriv_coo[canonical_idx] = deriv
        return third_deriv_coo

    def compute_sparse_derivatives(self, max_order: int = 3) -> List[Dict[Tuple[int, ...], sp.Expr]]:
        derivatives: List[Dict[Tuple[int, ...], sp.Expr]] = []
        if max_order >= 1:
            derivatives.append(self.compute_sparse_jacobian())
        if max_order >= 2:
            derivatives.append(self.compute_sparse_hessian())
        if max_order >= 3:
            derivatives.append(self.compute_sparse_third_derivatives())
        return derivatives

    def evaluate_at_steady_state(self, sparse_derivatives_list: List[Dict[Tuple[int, ...], sp.Expr]]) -> List[Dict[Tuple[int, ...], Any]]:
        values = (
            [self.model.symbolic.context["steady_states"].get(e, 0.0) for e in self.model.symbols["endogenous"]] * 3
            + [self.model.symbolic.context["steady_states"].get(e, 0.0) for e in self.model.symbols["exogenous"]]
        )
        values_dict = dict(zip(self.symbols, values))

        numerical_results = []
        for sparse_dict in sparse_derivatives_list:
            numerical_dict = {}
            for indices, expr in sparse_dict.items():
                try:
                    numerical_value = float(expr.subs(values_dict))
                    numerical_dict[indices] = numerical_value
                except (TypeError, ValueError):
                    numerical_dict[indices] = expr
            numerical_results.append(numerical_dict)

        return numerical_results
