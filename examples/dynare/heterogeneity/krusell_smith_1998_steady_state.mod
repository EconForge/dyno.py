/*
 * Krusell and Smith (1998)
 * Example: Computing the Steady State
 *
 * This example demonstrates:
 * 1. Computing the steady state numerically from an initial guess
 * 2. Solving the model
 * 3. Computing impulse response functions to aggregate shocks
 */

/*
 * Copyright © 2026 Dynare Team
 *
 * This file is part of Dynare.
 *
 * Dynare is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * Dynare is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with Dynare.  If not, see <https://www.gnu.org/licenses/>.
 */

// Declare heterogeneity dimension
heterogeneity_dimension households;

// Household-level variables
var(heterogeneity=households)
    c  (long_name = 'Consumption')
    a  (long_name = 'Assets')
;

// Household-level shocks
varexo(heterogeneity=households)
    e  (long_name = 'Idiosyncratic productivity shock')
;

// Aggregate variables
var
    Y (long_name = 'Aggregate output')
    r (long_name = 'Rate of return on capital net of depreciation')
    w (long_name = 'Wage rate')
    K (long_name = 'Aggregate capital')
;

// Aggregate shock
varexo eps_Z (long_name = 'Aggregate productivity shock');

// Parameters
parameters
    L     (long_name = 'Labor')
    alpha (long_name = 'Share of capital in production function')
    beta  (long_name = 'Subjective discount rate of households')
    delta (long_name = 'Capital depreciation rate')
    eis   (long_name = 'Elasticity of intertemporal substitution')
    Z_ss  (long_name = 'Aggregate TFP shock average value')
;

delta = 0.025;
eis = 1;
L = 1;
alpha = 0.11;
beta = 0.98;

verbatim;
r = 0.01;
rk = r + delta;
end;

% Normalize so that Y = 1
Z_ss = (rk / alpha) ^ alpha;

verbatim;
K = (alpha * Z_ss / rk) ^ (1 / (1 - alpha));
Y = Z_ss * K ^ alpha;
w = (1 - alpha) * Z_ss * (alpha * Z_ss / rk) ^ (alpha / (1 - alpha));

initial_guess = struct;
initial_guess.agg.w = w;
initial_guess.agg.r = r;
initial_guess.agg.Y = Y;
initial_guess.agg.K = K;

initial_guess.free_parameters.beta.initial_guess = 0.98;
initial_guess.free_parameters.beta.lower_bound = 0.97;
initial_guess.free_parameters.beta.upper_bound = 0.99;

rho_e = 0.966;
sig_e = 0.5;
[grid_e, ~, Pi_e] = rouwenhorst(rho_e, sig_e, 3, 1e-12, 1e5);
initial_guess.shocks.grids.e = grid_e;
initial_guess.shocks.Pi.e = Pi_e;

grid_a = logspace(log10(0.25), log10(200.25), 30)-0.25;
initial_guess.pol.grids.a = grid_a;
coh = (1+r)*grid_a + w*grid_e;
c = 0.1*coh;
a = coh-c;
initial_guess.pol.values.c = c;
initial_guess.pol.values.a = a;
initial_guess.pol.order = {'e', 'a'};
end;

// Household optimization problem
model(heterogeneity=households);
    [name='Euler equation with borrowing constraint']
    c^(-1/eis) = beta*(1+r(+1))*c(+1)^(-1/eis) ⟂ a>=0;

    [name='Budget constraint']
    c + a = (1+r)*a(-1) + w*e;
end;

// Aggregate equilibrium conditions
model;
    [name='Production function']
    Y = (Z_ss+eps_Z) * K(-1)^alpha * L^(1 - alpha);

    [name='Capital rental rate']
    r = alpha * (Z_ss+eps_Z) * (K(-1) / L)^(alpha - 1) - delta;

    [name='Wage rate']
    w = (1 - alpha) * (Z_ss+eps_Z) * (K(-1) / L)^alpha;

    [name='Capital market clearing']
    K = SUM(a);
end;

// Aggregate shock specification
shocks;
    var eps_Z; stderr 0.01;
end;

//==========================================================================
// STEP 1: Compute steady state from initial guess
//==========================================================================
// The initial guess (policy functions, grids, shock discretization)
// is constructed in the verbatim block above. The discount factor beta is
// calibrated so that the capital market clearing condition holds.
heterogeneity_compute_steady_state(variable = initial_guess);

//==========================================================================
// STEP 2: Solve the model
//==========================================================================
// Compute the linearized solution using sequence-space Jacobians
// truncation_horizon = 400 specifies the time horizon for fake news algorithm
heterogeneity_solve(truncation_horizon = 400);

//==========================================================================
// STEP 3: Compute impulse response functions
//==========================================================================
// Compute IRFs with 80 periods horizon
heterogeneity_simulate(irf = 80);
