/*
 * Two-Asset HANK Model (Heterogeneous Agent New Keynesian)
 * Example: Computing the Steady State with Multi-Parameter Calibration
 *
 * This example demonstrates:
 * 1. Computing the steady state numerically from an initial guess
 * 2. Calibrating multiple free parameters (beta_ss, vphi, chi1) to match
 *    three market-clearing conditions simultaneously
 * 3. Solving a HANK model with liquid and illiquid assets
 * 4. Simulating responses to anticipated shock sequences (news shocks)
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
   b   (long_name = 'liquid assets (bonds)')
   a   (long_name = 'illiquid assets (equity)')
   c   (long_name = 'consumption')
   Va  (long_name = 'derivative of value function w.r.t. a')
   Vb  (long_name = 'derivative of value function w.r.t. b')
   u   (long_name = 'effective labor')
;

// Household-level shocks
varexo(heterogeneity=households)
   e   (long_name = 'idiosyncratic efficiency')
;

// Aggregate variables
var
    piw    (long_name = 'wage inflation')
    psiw   (long_name = 'wage adjustment cost')
    rb     (long_name = 'return on bonds (liquid)')
    ra     (long_name = 'return on equity (illiquid)')
    tax    (long_name = 'tax rate')
    i      (long_name = 'nominal interest rate')
    psip   (long_name = 'price adjustment cost')
    I      (long_name = 'investment')
    Q      (long_name = 'Tobin Q')
    w      (long_name = 'real wage')
    N      (long_name = 'aggregate labor')
    K      (long_name = 'capital')
    div    (long_name = 'dividends')
    p      (long_name = 'equity price')
    pi     (long_name = 'inflation')
    mc     (long_name = 'marginal cost')
    r      (long_name = 'real interest rate')
    Y      (long_name = 'output')
;

// Aggregate shocks
varexo
    rstar       (long_name = 'monetary policy shock')
    markup      (long_name = 'price markup shock')
    G           (long_name = 'government spending shock')
    beta        (long_name = 'discount factor shock')
    Z           (long_name = 'productivity shock')
    rinv_shock  (long_name = 'investment-specific shock')
    markup_w    (long_name = 'wage markup shock')
;

// Parameters
parameters
    kappap alpha epsI muw phi omega Bg Bh pshare delta
    kappaw frisch mup vphi eis
    chi0 chi1 chi2
    Z_ss beta_ss r_ss G_ss
;

Bg = 2.8;
Bh = 1.04;
G_ss = 0.2;
chi0 = 0.25;
chi2 = 2;
delta = 0.02;
eis = 0.5;
epsI = 4;
frisch = 1;
kappap = 0.1;
kappaw = 0.1;
muw = 1.1;
omega = 0.005;
phi = 1.5;
r_ss = 0.0125;

verbatim;
tot_wealth = 14;
K_ss = 10;
p_ss = tot_wealth - Bg;
mc_ss = 1 - r_ss * (p_ss - K_ss);
end;

mup = 1 / mc_ss;
alpha = (r_ss + delta) * K_ss / mc_ss;
Z_ss = K_ss ^ (-alpha);
pshare = p_ss / (tot_wealth - Bh);

verbatim;
w = mc_ss * (1 - alpha);
tax = (r_ss * Bg + G_ss) / w;
I = delta * K_ss;
div = 1 - w - I;
rb = r_ss - omega;
ra = r_ss;

initial_guess = struct;
initial_guess.agg.piw = 0;
initial_guess.agg.psiw = 0;
initial_guess.agg.rb = rb;
initial_guess.agg.ra = ra;
initial_guess.agg.tax = tax;
initial_guess.agg.i = r_ss;
initial_guess.agg.psip = 0;
initial_guess.agg.I = I;
initial_guess.agg.Q = 1;
initial_guess.agg.w = w;
initial_guess.agg.N = 1;
initial_guess.agg.K = K_ss;
initial_guess.agg.div = div;
initial_guess.agg.p = p_ss;
initial_guess.agg.pi = 0;
initial_guess.agg.mc = mc_ss;
initial_guess.agg.r = r_ss;
initial_guess.agg.Y = 1;

ne = 3;
rho_e = 0.966;
sig_e = 0.92;
[grid_e, ~, Pi_e] = rouwenhorst(rho_e, sig_e, ne, 1e-12, 1e5);
initial_guess.shocks.grids.e = grid_e;
initial_guess.shocks.Pi.e = Pi_e;

nb = 10;
na = 20;
a_max = 4000;
b_max = 50;
grid_a = logspace(log10(0.25), log10(a_max+0.25), na)-0.25;
grid_b = logspace(log10(0.25), log10(b_max+0.25), nb)-0.25;
initial_guess.pol.grids.a = grid_a;
initial_guess.pol.grids.b = grid_b;
initial_guess.pol.order = {'e', 'b', 'a'};

grid_a_3d = reshape(grid_a, 1, 1, []);
grid_b_3d = reshape(grid_b, 1, [], 1);
grid_e_3d = reshape(grid_e, [], 1, 1);
coh = (1+rb)*grid_b_3d+(1+ra)*grid_a_3d+(1-tax)*w*grid_e_3d;
b = grid_b_3d .* ones(ne, nb, na);
a = (1+ra) * grid_a_3d .* ones(ne, nb, na);
c = max(coh - a - b, 1e-8);
Vb = (1+rb) * (c .^ (-1/eis));
Va = (1+ra) * (c .^ (-1/eis));
u = grid_e_3d .* (c .^ (-1/eis));

initial_guess.pol.values.Va = Va;
initial_guess.pol.values.Vb = Vb;
initial_guess.pol.values.a = a;
initial_guess.pol.values.b = b;
initial_guess.pol.values.c = c;
initial_guess.pol.values.u = u;

initial_guess.free_parameters.beta_ss.initial_guess = 0.97;
initial_guess.free_parameters.beta_ss.lower_bound = 0.01;
initial_guess.free_parameters.beta_ss.upper_bound = 0.999;
initial_guess.free_parameters.chi1.initial_guess = 6.4;
initial_guess.free_parameters.chi1.lower_bound = 0.01;
initial_guess.free_parameters.vphi.initial_guess = 1.7;
initial_guess.free_parameters.vphi.lower_bound = 0.01;
end;

// Household optimization problem with two assets
model(heterogeneity=households);
   [name='Euler equation for liquid assets']
   c^(-1/eis) - (beta_ss+beta) * Vb(+1) = 0 ⟂ b >= 0;

   [name='Euler equation for illiquid assets']
   c^(-1/eis)*(1 + chi1 * sign(a-(1+ra)*a(-1)) * (abs(a-(1+ra)*a(-1))/((1+ra)*a(-1)+chi0))^(chi2-1)) - (beta_ss+beta) * Va(+1) = 0 ⟂ a >= 0;

   [name='Budget constraint with adjustment costs']
   (1 + ra) * a(-1) + (1 + rb) * b(-1) - (chi1 / chi2) * abs(a-(1+ra)*a(-1))^chi2 * ((1+ra)*a(-1)+chi0)^(1-chi2) + (1-tax) * w * N * e - c - a - b;

   [name='Effective labor']
   u = e * c^(-1/eis);

   [name='Envelope condition for illiquid assets']
   Va = (1 + ra)*(1 - (chi1/chi2) * ( - chi2 * sign(a-(1+ra)*a(-1)) * (abs(a-(1+ra)*a(-1))/((1+ra)*a(-1)+chi0))^(chi2-1) + (1-chi2) * (abs(a-(1+ra)*a(-1))/((1+ra)*a(-1)+chi0))^chi2 )) * c^(-1/eis);

   [name='Envelope condition for liquid assets']
   Vb = (1 + rb) * c^(-1/eis);
end;

// Aggregate equilibrium conditions
model;
   [name='New Keynesian Phillips curve']
   kappap * (mc - 1 / mup) + Y(+1) / Y * log(1 + pi(+1)) / (1 + r(+1)) + markup - log(1 + pi);

   [name='Equity pricing']
   div(+1) + p(+1) - p * (1 + r(+1));

   [name='Production function']
   N = (Y / (Z_ss+Z) / K(-1) ^ alpha) ^ (1 / (1 - alpha));

   [name='Labor demand']
   mc = w * N / (1 - alpha) / Y;

   [name='Tobin Q']
   (K / K(-1) - 1) / (delta * epsI) + 1 - Q;

   [name='Valuation equation']
   alpha * (Z_ss+Z(+1)) * (N(+1) / K) ^ (1 - alpha) * mc(+1) - (K(+1) / K -
   (1 - delta) + (K(+1) / K - 1) ^ 2 / (2 * delta * epsI)) + K(+1) / K * Q(+1) - (1 + r(+1) + rinv_shock) * Q;

   [name='Price adjustment cost']
   mup / (mup - 1) / 2 / kappap * log(1 + pi) ^ 2 * Y - psip;

   [name='Capital accumulation']
   K - (1 - delta) * K(-1) + K(-1) * (K / K(-1) - 1) ^ 2 / (2 * delta * epsI) - I;

   [name='Taylor rule']
   rstar + r_ss + phi * pi - i;

   [name='Government budget constraint']
   (r * Bg + G_ss + G) / w / N - tax;

   [name='Return on liquid assets']
   r - omega - rb;

   [name='Return on illiquid assets']
   pshare * (div + p) / p(-1) + (1 - pshare) * (1 + r) - 1 - ra;

   [name='Fisher equation']
   1 + i(-1) - (1 + r) * (1 + pi);

   [name='Wage inflation']
   (1 + pi) * w / w(-1) - 1 - piw;

   [name='Wage adjustment cost']
   muw / (1 - muw) / 2 / kappaw * log(1 + piw) ^ 2 * N - psiw;

   [name='Wage Phillips curve']
   kappaw * (vphi * N ^ (1 + 1 / frisch) - (1 - tax) * w * N * SUM(u) / muw) + (beta_ss+beta) * log(1 + piw(+1)) + markup_w - log(1 + piw);

   [name='Illiquid asset market clearing']
   p + Bg - SUM(a) - SUM(b);

   [name='Liquid asset market clearing']
   Bh - SUM(b);
end;

// News shock sequence
// Anticipated shocks known at t=0
shocks;
    var rstar;
    periods 1;
    values 0.01;

    var G;
    periods 1:10;
    values 0.02;

    var markup;
    periods 5:8;
    values 0.005;
end;

//==========================================================================
// STEP 1: Compute steady state with multi-parameter calibration
//==========================================================================
// The initial guess is constructed in the verbatim block above. Three
// parameters (beta_ss, vphi, chi1) are calibrated so that the wage Phillips
// curve and two asset market clearing conditions hold.
heterogeneity_compute_steady_state(variable = initial_guess,
    calibration_target_equations=['Wage Phillips curve',
        'Liquid asset market clearing',
        'Illiquid asset market clearing'],
    time_iteration_tol=1e-10,
    time_iteration_max_iter=2000,
    time_iteration_early_stopping=0,
    time_iteration_solver_tolf=1e-12,
    time_iteration_solver_tolx=1e-14);

//==========================================================================
// STEP 2: Solve the model
//==========================================================================
// Compute the linearized solution using sequence-space Jacobians
heterogeneity_solve(truncation_horizon = 300);

//==========================================================================
// STEP 3: Simulate with news shock sequence
//==========================================================================
// When anticipated shocks are defined in the shocks block (with periods/values),
// heterogeneity_simulate automatically uses news shock mode
heterogeneity_simulate;
