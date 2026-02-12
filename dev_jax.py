from dyno.symbolic_model import DynoModel

model = DynoModel("examples/modfiles/RBC.mod")
r0, A0, B0, C0, D0 = model.jacobians  # for comparison purpose

import jax

jax.config.update("jax_enable_x64", True)
from jax import numpy as jnp


evaluator = model.data.evaluator
evaluator.function_table["log"] = jnp.log
evaluator.function_table["exp"] = jnp.exp
evaluator.function_table["sqrt"] = jnp.sqrt
evaluator.function_table["abs"] = jnp.abs
evaluator.function_table["pow"] = jnp.pow

from dyno.dynsym.grammar import (
    stringify_symbol,
    stringify_variable,
    stringify_value,
    stringify_constant,
)

model.data.equations


y_, e_ = model.__steady_state_vectors__

y_0 = jnp.array(y_)
y_1 = jnp.array(y_)
y_2 = jnp.array(y_)
e_ = jnp.array(e_)


def eval_equations(y_0, y_1, y_2, e_):

    v_context = model.data.context["variables"]
    for i, v in enumerate(model.symbols["endogenous"]):
        v_context[v] = {
            -1: y_2[i],
            0: y_1[i],
            1: y_0[i],
        }
    for i, v in enumerate(model.symbols["exogenous"]):
        v_context[v] = {
            0: e_[i],
        }

    res = [model.data.evaluator.visit(eq) for eq in model.data.equations]
    # return res
    residuals = jnp.array(res)
    return residuals
    # return residuals


import time

t1 = time.time()
r1 = eval_equations(y_0, y_1, y_2, e_)
t2 = time.time()
print("Elapsed: ", t2 - t1)

j_eval_equations = jax.jit(eval_equations)
r2 = j_eval_equations(y_0, y_1, y_2, e_)

t1 = time.time()
r2 = j_eval_equations(y_0, y_1, y_2, e_)

t2 = time.time()
print("Elapsed (jitted): ", t2 - t1)


def eval_equations_with_jacs(y_0, y_1, y_2, e_):
    r = eval_equations(y_0, y_1, y_2, e_)
    A = jax.jacobian(lambda u: eval_equations(u, y_1, y_2, e_))(y_0)
    B = jax.jacobian(lambda u: eval_equations(y_0, u, y_2, e_))(y_1)
    C = jax.jacobian(lambda u: eval_equations(y_0, y_1, u, e_))(y_2)
    D = jax.jacobian(lambda u: eval_equations(y_0, y_1, y_2, u))(e_)
    return r, A, B, C, D


eval_equations_with_jacs_jit = jax.jit(eval_equations_with_jacs)

r, A, B, C, D = eval_equations_with_jacs_jit(y_0, y_1, y_2, e_)

t1 = time.time()
r, A, B, C, D = eval_equations_with_jacs_jit(y_0, y_1, y_2, e_)
t2 = time.time()
print("Elapsed (jitted with jacs): ", t2 - t1)


N = 100000
v_0 = jnp.repeat(y_0[None, :], N, axis=0)
v_1 = jnp.repeat(y_1[None, :], N, axis=0)
v_2 = jnp.repeat(y_2[None, :], N, axis=0)
v_e = jnp.repeat(e_[None, :], N, axis=0)

jax.vmap(eval_equations_with_jacs_jit, in_axes=(0, 0, 0, 0))(
    v_0, v_1, v_2, v_e
)  # This works !!!

vec_eval_equations_with_jacs_jit = jax.jit(
    jax.vmap(eval_equations_with_jacs_jit, in_axes=(0, 0, 0, 0))
)

# compile
vec_eval_equations_with_jacs_jit(v_2, v_1, v_0, v_e)

t1 = time.time()
v_r, v_A, v_B, v_C, v_D = vec_eval_equations_with_jacs_jit(v_0, v_1, v_2, v_e)
t2 = time.time()
print("Elapsed (jitted with jacs): ", t2 - t1)


# no check the results make sense...


print(jnp.abs(v_A[50, :, :] - A0).max())
print(jnp.abs(v_B[50, :, :] - B0).max())
print(jnp.abs(v_C[50, :, :] - C0).max())
