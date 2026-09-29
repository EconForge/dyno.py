/*
 * The file shows how to simulate a one-time TFP shock in a basic RBC model
 * using the perfect foresight solver.
 *
 * The model features a closed economy with a representative having the CRRA felicity
 * function c^(1-sigma)/(1-sigma)', where 'c' is consumption and 'sigma' is
 * relative risk aversion. The subjective discount factor is 'beta'. Labor is in fixed
 * supply, i.e. equal to 1.
 *
 * Production employs a Cobb-Douglas function 'z*k(-1)^alpha', where 'z' is a
 * stochastic technology level variable, 'k' is capital (using Dynare's
 * default end-of-period timing convention), and 'alpha' measures the capital share.
 *
 * The capital stock evolves according to the usual law of motion, where 'delta'
 * is the depreciation rate.
 *
 * - initval, followed by the steady command, is used to set the initial and terminal
 *      condition for the simulation conditional on the provided value for TFP
        z.
 * - a shocks-block sets a one-time TFP shock in the first period
 * - rplot is used to plot the simulation results
 */

/*
 * Copyright © 2001-2025 Dynare Team
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


// Endogenous variables: consumption and capital
var c k;

// Exogenous variable: technology level
varexo z;

// Parameters declaration and calibration
parameters alpha     ${\alpha}$  (long_name='capital elasticity of output')
           sigma     ${\sigma}$  (long_name='risk aversion')
           delta     ${\delta}$  (long_name='depreciation rate')
           beta      ${\beta}$   (long_name='time preference rate')
           ;

alpha=0.5;
sigma=0.5;
delta=0.02;
beta=1/(1+0.05);

// Equilibrium conditions
model;
[name='Resource constraint']
c + k = z*k(-1)^alpha + (1-delta)*k(-1);
[name='Euler equation']
c^(-sigma) = beta*(alpha*z(+1)*k^(alpha-1) + 1-delta)*c(+1)^(-sigma);
end;

steady_state_model;
k = ((1/beta-(1-delta))/(z*alpha))^(1/(alpha-1));
c = z*k^alpha-delta*k;
end;

//set initial steady state
initval;
z=1;
end;
steady; //compute steady state conditional on value of z in initval

// Check the Blanchard-Kahn conditions
check;

// Declare a positive technological shock in period 1
shocks;
var z;
periods 1;
values 1.2;
end;

// Prepare the deterministic simulation of the model over 200 periods
perfect_foresight_setup(periods=200);

// Perform the simulation
perfect_foresight_solver;

// Display the path of consumption and capital
rplot c;
rplot k;
