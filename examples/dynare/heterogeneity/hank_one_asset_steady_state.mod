/*
 * One-Asset HANK Model (Heterogeneous Agent New Keynesian)
 * Example: Computing the Steady State with Parameter Calibration
 *
 * This example demonstrates:
 * 1. Computing the steady state numerically from an initial guess
 * 2. Calibrating a free parameter (beta) to match a market-clearing condition
 * 3. Solving a HANK model with one liquid asset
 * 4. Running stochastic simulations to generate time series
 *
 * The model equations and calibration are taken from Auclert, Bardóczy,
 * Rognlie and Straub (2021): “Using the Sequence-Space Jacobian to Solve and
 * Estimate Heterogeneous-Agent Models” (see the manuscript rather than the
 * published version for the details).
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
   c      (long_name = 'consumption')
   n      (long_name = 'labor supply')
   ns     (long_name = 'effective labor supply')
   a      (long_name = 'assets')
;

// Household-level shocks
varexo(heterogeneity=households)
   e      (long_name = 'idiosyncratic efficiency')
;

// Aggregate variables
var
   Y    (long_name = 'Aggregate output')
   L    (long_name = 'Aggregate labor')
   w    (long_name = 'Real wage')
   pi   (long_name = 'Inflation')
   Div  (long_name = 'Dividends')
   Tax  (long_name = 'Taxes')
   r    (long_name = 'Real interest rate')
;

// Aggregate shocks
varexo G markup rstar;

// Parameters
parameters
   beta vphi
   eis frisch
   mu kappa phi
   Z B r_ss
;

B = 5.6;
Z = 1;
eis = 0.5;
frisch = 0.5;
kappa = 0.1;
mu = 1.2;
phi = 1.5;
r_ss = 0.005;

verbatim;
w = 1/mu;
Div = 1-w;
Tax = r_ss*B;

initial_guess = struct;
initial_guess.agg.Y = 1;
initial_guess.agg.L = 1;
initial_guess.agg.w = w;
initial_guess.agg.pi = 0;
initial_guess.agg.Div = Div;
initial_guess.agg.Tax = Tax;
initial_guess.agg.r = r_ss;

rho_e = 0.966;
sig_e = 0.5;
[grid_e, ~, Pi_e] = rouwenhorst(rho_e, sig_e, 3, 1e-12, 1e5);
initial_guess.shocks.grids.e = grid_e;
initial_guess.shocks.Pi.e = Pi_e;

grid_a = logspace(log10(0.25), log10(200.25), 30)-0.25;
initial_guess.pol.grids.a = grid_a;

T = (Div-Tax)*grid_e;
fininc = (1+r_ss)*grid_a+T;

coh = (1+r_ss)*grid_a+w*grid_e+T;
c = 0.1*coh;
a = coh-c;
n = ones(size(a));
ns = n .* grid_e;

initial_guess.pol.values.c = c;
initial_guess.pol.values.a = a;
initial_guess.pol.values.n = n;
initial_guess.pol.values.ns = ns;

initial_guess.free_parameters.beta.initial_guess = 0.98;
initial_guess.free_parameters.beta.upper_bound = 0.999;
initial_guess.free_parameters.beta.lower_bound = 0;

initial_guess.free_parameters.vphi.initial_guess = 0.78;
initial_guess.free_parameters.vphi.lower_bound = 0.01;

initial_guess.pol.order = {'e', 'a'};
end;

// Household optimization problem
model(heterogeneity=households);
   [name='Euler equation with borrowing constraint']
   c^(-1/eis) - beta * (1 + r(+1)) * c(+1)^(-1/eis) = 0 ⟂ a >= 0;

   [name='Budget constraint']
   (1 + r) * a(-1) + w * n * e + (Div-Tax) * e - c - a;

   [name='Labor supply']
   vphi*n^(1/frisch) - w*e*c^(-1/eis);

   [name='Effective labor supply']
   ns = n * e;
end;

// Aggregate equilibrium conditions
model;
   [name='Labor demand']
   L - Y / Z;

   [name='Dividends']
   Div - (Y - w * L - mu / (mu - 1) / (2 * kappa) * log(1 + pi)^2 * Y);

   [name='Taylor rule']
   (1 + r_ss + rstar(-1) + phi * pi(-1)) / (1 + pi) - 1 - r;

   [name='Government budget constraint']
   Tax - (r * B) - G;

   [name='New Keynesian Phillips curve']
   kappa * (w / Z - 1 / mu)
   + Y(+1)/Y * log(1 + pi(+1)) / (1 + r(+1))
   + markup
   - log(1 + pi);

   [name='Asset market clearing']
   SUM(a) - B;

   [name='Labor market clearing']
   sum(ns) - L;
end;

// Aggregate shock specification
shocks;
    var G; stderr 0.01;
    var markup; stderr 0.01;
    var rstar; stderr 0.01;
end;

//==========================================================================
// STEP 1: Compute steady state with parameter calibration
//==========================================================================
// The initial guess is constructed in the verbatim block above. The discount
// factor beta and vphi are calibrated so that the asset and labor market
// clearing conditions hold.
heterogeneity_compute_steady_state(variable = initial_guess,
    calibration_target_equations=['Asset market clearing', 'Labor market clearing'],
    time_iteration_solver_stop_on_error);

//==========================================================================
// STEP 2: Solve the model
//==========================================================================
// Compute the linearized solution using sequence-space Jacobians
heterogeneity_solve(truncation_horizon = 300);

//==========================================================================
// STEP 3: Run stochastic simulation
//==========================================================================
// Simulate 1000 periods
heterogeneity_simulate(periods = 1000);
