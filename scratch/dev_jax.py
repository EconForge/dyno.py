from dyno.dyno_model import DynoModel
from dyno import examples_path

model = DynoModel(examples_path("modfiles", "RBC.mod"))
r0, A0, B0, C0, D0 = model.jacobians  # for comparison purpose

import jax
jax.config.update("jax_enable_x64", True)
from jax import numpy as jnp

from dyno.experimental.jax import build_residual_function, build_residual_with_jacs_function

y_, e_ = model.__steady_state_vectors__

y_0 = jnp.array(y_)
y_1 = jnp.array(y_)
y_2 = jnp.array(y_)
e_ = jnp.array(e_)

eval_equations = build_residual_function(model)

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


eval_equations_with_jacs = build_residual_with_jacs_function(model)
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
vec_eval_equations_with_jacs_jit(v_0, v_1, v_2, v_e)

t1 = time.time()
v_r, v_A, v_B, v_C, v_D = vec_eval_equations_with_jacs_jit(v_0, v_1, v_2, v_e)
t2 = time.time()
print("Elapsed (vectorized and jitted with jacs): ", t2 - t1)


# now check the results make sense...

print(jnp.abs(v_A[50, :, :] - A0).max())
print(jnp.abs(v_B[50, :, :] - B0).max())
print(jnp.abs(v_C[50, :, :] - C0).max())

