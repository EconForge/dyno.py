from typing import Callable, Any
from dyno.dyno_model import DynoModel

def build_residual_function(model: DynoModel) -> Callable:
    """
    Build a pure JAX-compatible residual function for the given DynoModel.

    Args:
        model: A DynoModel instance.
        
    Returns:
        res_fn: A pure Python function `res_fn(y_plus, y_curr, y_minus, e_curr)`
                that returns the residuals of the model's equations.
    """
    import jax
    import jax.numpy as jnp
    from dyno.dynsym.analyze import EquationsEvaluator

    endogenous = model.symbols["endogenous"]
    exogenous = model.symbols["exogenous"]
    equations = model.symbolic.equations
    base_constants = model.symbolic.context.get("constants", {})

    def res_fn(y_plus, y_curr, y_minus, e_curr):
        v_context = {}
        for i, v in enumerate(endogenous):
            v_context[v] = {
                -1: y_minus[i],
                0: y_curr[i],
                1: y_plus[i],
            }
        for i, v in enumerate(exogenous):
            v_context[v] = {
                0: e_curr[i],
            }

        context = {
            "variables": v_context,
            "constants": base_constants,
        }

        # Initialize a temporary evaluator specific to this execution
        evaluator = EquationsEvaluator(
            context=context,
        )

        # Update the function table with JAX functions after initialization
        # so it overrides autodiff.MATH_FUNCTIONS
        evaluator.function_table.update({
            "log": jnp.log,
            "exp": jnp.exp,
            "sqrt": jnp.sqrt,
            "abs": jnp.abs,
            "pow": jnp.pow,
        })

        res = [evaluator.visit(eq) for eq in equations]
        return jnp.array(res)

    return res_fn


def build_residual_with_jacs_function(model: DynoModel) -> Callable:
    """
    Build a pure JAX-compatible function that computes residuals and Jacobians.

    Args:
        model: A DynoModel instance.
        
    Returns:
        res_with_jacs_fn: A function `fn(y_plus, y_curr, y_minus, e_curr)` returning
                          `(residuals, A, B, C, D)` where:
                          A is the Jacobian with respect to y_plus
                          B is the Jacobian with respect to y_curr
                          C is the Jacobian with respect to y_minus
                          D is the Jacobian with respect to e_curr
    """
    import jax
    res_fn = build_residual_function(model)

    def eval_equations_with_jacs(y_plus, y_curr, y_minus, e_curr):
        r = res_fn(y_plus, y_curr, y_minus, e_curr)
        A = jax.jacobian(lambda u: res_fn(u, y_curr, y_minus, e_curr))(y_plus)
        B = jax.jacobian(lambda u: res_fn(y_plus, u, y_minus, e_curr))(y_curr)
        C = jax.jacobian(lambda u: res_fn(y_plus, y_curr, u, e_curr))(y_minus)
        D = jax.jacobian(lambda u: res_fn(y_plus, y_curr, y_minus, u))(e_curr)
        return r, A, B, C, D

    return eval_equations_with_jacs
