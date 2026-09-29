/*
 * An elementary RBC model (same as perfect_foresight_rbc.mod), simulated in perfect
 * foresight with expectation errors: agents behave as under perfect foresight, but they
 * can still be surprised by unexpected shocks, and thus recompute their
 * optimal plans when such an unexpected shock happens.
 *
 * - initval, followed by the steady command, is used to set the initial and terminal
 *      condition for the simulation conditional on the provided value for TFP
        z for period 1.
 * - a shocks-block sets a one-time TFP shock in the first period
 * - shocks(learnt_in = period) blocks are used to set surprise shocks learned in later periods
 * - an endval(learnt_in = period) is used to set a permanent shock and the
 *      associated terminal condition
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

// Declare a positive technological shock in period 3, that agents only learn about in period 2
shocks(learnt_in = 2);
  var z;
  periods 3;
  values 1.1;
end;

// Declare a permanent positive technological shock, that agents only learn about in period 5.
// Note that this will automatically trigger the recomputation of the terminal steady state.
endval(learnt_in = 5);
  z = 1.05;
end;

// Prepare the deterministic simulation of the model over 200 periods
perfect_foresight_with_expectation_errors_setup(periods=200);

// Perform the simulation
perfect_foresight_with_expectation_errors_solver;

// Display the path of consumption and capital
rplot c;
rplot k;
