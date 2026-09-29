/*
 * This file shows how to solve an RBC model with two occasionally binding constraints:
 *  1. The INEG constraint implements quadratic capital adjustment costs if investment
 *      falls below its steady state. If investment is above steady state, there are no
 *      adjustment costs
 *  2. The IRR constraint implements irreversible investment. Investment cannot be lower
 *      than a factor phi of its steady state.
 *
 * Notes:
 *  - This mod-file is based on an example originally provided by Luca Guerrieri
 *      and Matteo Iacoviello provided at https://www.matteoiacoviello.com/research_files/occbin_20140630.zip
 *  - The INEG constraint should theoretically be log_Invest-log(steady_state(Invest))<0, but this will lead
 *      to numerical issues. Instead we allow for a small negative value of <-0.000001
 *
 * Please note that the following copyright notice only applies to this Dynare
 * implementation of the model.
 */

/*
 * Copyright © 2021 Dynare Team
 *
 * This is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * It is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * For A copy of the GNU General Public License,
 * see <https://www.gnu.org/licenses/>.
 */


var A           $A$         (long_name='TFP')
    C           $C$         (long_name='consumption')
    Invest      $I$         (long_name='investment')
    K           $K$         (long_name='capital')
    Lambda      $\lambda$   (long_name='Lagrange multiplier')
    log_K       ${\hat K}$  (long_name='log capital')
    log_Invest  ${\hat I}$  (long_name='log investment')
    log_C       ${\hat C}$  (long_name='log consumption')
    ;

varexo epsilon $\varepsilon$        (long_name='TFP shock');

parameters alpha    $\alpha$        (long_name='capital share')
        delta       $\delta$        (long_name='depreciation')
        beta        $\beta$         (long_name='discount factor')
        sigma       $\sigma$        (long_name='risk aversion')
        rho         $\rho$          (long_name='autocorrelation TFP')
        phi         $\phi$          (long_name='irreversibility fraction of steady state investment')
        psi         $\psi$          (long_name='capital adjustment cost')
        ;

beta=0.96;
alpha=0.33;
delta=0.10;
sigma=2;
rho = 0.9;
phi = 0.975;
psi = 5;


model;
// 1.
[name='Euler', bind = 'INEG']
-C^(-sigma)*(1+2*psi*(K/K(-1)-1)/K(-1))+ beta*C(+1)^(-sigma)*((1-delta)-2*psi*(K(+1)/K-1)*
  (-K(+1)/K^2)+alpha*exp(A(+1))*K^(alpha-1))= -Lambda+beta*(1-delta)*Lambda(+1);

[name='Euler', relax = 'INEG']
-C^(-sigma) + beta*C(+1)^(-sigma)*(1-delta+alpha*exp(A(+1))*K^(alpha-1))= -Lambda+beta*(1-delta)*Lambda(+1);

// 2.
[name='Budget constraint',bind = 'INEG']
C+K-(1-delta)*K(-1)+psi*(K/K(-1)-1)^2=exp(A)*K(-1)^(alpha);

[name='Budget constraint',relax = 'INEG']
C+K-(1-delta)*K(-1)=exp(A)*K(-1)^(alpha);

// 3.
[name='LOM capital']
Invest = K-(1-delta)*K(-1);

// 4.
[name='investment',bind='IRR,INEG']
(log_Invest - log(phi*steady_state(Invest))) = 0;
[name='investment',relax='IRR']
Lambda=0;
[name='investment',bind='IRR',relax='INEG']
(log_Invest - log(phi*steady_state(Invest))) = 0;

// 5.
[name='LOM TFP']
A = rho*A(-1)+epsilon;

// Definitions
[name='Definition log capital']
log_K=log(K);
[name='Definition log consumption']
log_C=log(C);
[name='Definition log investment']
log_Invest=log(Invest);
end;

occbin_constraints;
name 'IRR'; bind log_Invest-log(steady_state(Invest))<log(phi); relax Lambda<0;
name 'INEG'; bind log_Invest-log(steady_state(Invest))<-0.000001; %not exactly 0 for numerical reasons
end;

steady_state_model;
K = ((1/beta-1+delta)/alpha)^(1/(alpha-1));
C = -delta*K +K^alpha;
Invest = delta*K;
log_K = log(K);
log_C = log(C);
log_Invest = log(Invest);
Lambda = 0;
A=0;
end;

shocks;
  var epsilon; stderr 0.015;
end;

steady;

shocks(surprise);
var epsilon;
periods 1:9, 10, 50, 90, 130, 131:169;
values -0.0001, -0.01,-0.02, 0.01, 0.02, 0;
end;

occbin_setup;
occbin_solver(simul_periods=200,simul_check_ahead_periods=200);

occbin_graph log_C epsilon Lambda log_K log_Invest A;
