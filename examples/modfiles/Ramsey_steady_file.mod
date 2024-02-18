/*
 * This file replicates the model studied in:
 * Lawrence J. Christiano, Roberto Motto and Massimo Rostagno (2007):
 * "Notes on Ramsey-Optimal Monetary Policy", Section 2
 * The paper is available at http://faculty.wcas.northwestern.edu/~lchrist/d16/d1606/ramsey.pdf
 * 
 * Notes:
 * - This mod-files allows to simulate a simple New Keynesian Model with Rotemberg price 
 *      adjustment costs under fully optimal monetary under commitment (Ramsey)
 *
 *  - This files shows how to use a user-defined conditional steady state file in the Ramsey case. It takes
 *      the value of the defined instrument R as given and then computes the rest of the steady 
 *      state, including the steady state inflation rate, based on this value. The initial value 
 *      of the instrument for steady state search must then be defined in an initval-block.
 *
 * This implementation was written by Johannes Pfeifer.
 *
 * If you spot mistakes, email me at jpfeifer@gmx.de
 *
 * Please note that the following copyright notice only applies to this Dynare 
 * implementation of the model.
 */

/*
 * Copyright © 2019-2022 Dynare Team
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
 * For a copy of the GNU General Public License,
 * see <https://www.gnu.org/licenses/>.
 */


var C       $C$                 (long_name='Consumption')
    pi      $\pi$               (long_name='Gross inflation')
    h       $h$                 (long_name='hours worked')
    Z       $Z$                 (long_name='TFP')
    R       $R$                 (long_name='Net nominal interest rate')
    log_C   ${\ln C}$           (long_name='Log Consumption')
    log_h   ${\ln h}$           (long_name='Log hours worked')
    pi_ann  ${\pi^{ann}}$       (long_name='Annualized net inflation')
    R_ann   ${R^{ann}}$         (long_name='Annualized net nominal interest rate')
    r_real  ${r^{ann,real}}$    (long_name='Annualized net real interest rate')
    y_nat   ${y^{nat}}$         (long_name='Natural (flex price) output')
    y_gap   ${r^{gap}}$         (long_name='Output gap')
;

varexo epsilon ${\varepsilon}$     (long_name='TFP shock')
    ;

parameters beta     ${\beta}$       (long_name='discount factor')
        theta       ${\theta}$      (long_name='substitution elasticity')
        tau         ${\tau}$        (long_name='labor subsidy')
        chi         ${\chi}$        (long_name='labor disutility')
        phi         ${\phi}$        (long_name='price adjustment costs')
        rho         ${\rho}$        (long_name='TFP autocorrelation')
        ;

beta=0.99;
theta=5;
phi=100;
rho=0.9;
tau=0;
chi=1;
        
model;
    [name='Euler equation']
    1/(1+R)=beta*C/(C(+1)*pi(+1));
    [name='Firm FOC']
    (tau-1/(theta-1))*(1-theta)+theta*(chi*h*C/(exp(Z))-1)=phi*(pi-1)*pi-beta*phi*(pi(+1)-1)*pi(+1);
    [name='Resource constraint']
    C*(1+phi/2*(pi-1)^2)=exp(Z)*h;
    [name='TFP process']
    Z=rho*Z(-1)+epsilon;
    [name='Definition log consumption']
    log_C=log(C);
    [name='Definition log hours worked']
    log_h=log(h);
    [name='Definition annualized inflation rate']
    pi_ann=4*log(pi);
    [name='Definition annualized nominal interest rate']
    R_ann=4*R;
    [name='Definition annualized real interest rate']
    r_real=4*log((1+R)/pi(+1));
    [name='Definition natural output']
    y_nat=exp(Z)*sqrt((theta-1)/theta*(1+tau)/chi);
    [name='output gap']
    y_gap=log_C-log(y_nat);
end;

initval;
    R=1/beta-1;
end;

shocks;
    var epsilon = 0.01^2;
end;

//use Ramsey optimal policy

//define planner objective, which corresponds to utility function of agents
planner_objective log(C)-chi/2*h^2;

//set up Ramsey optimal policy problem with interest rate R as the instrument,...
// defining the discount factor in the planner objective to be the one of private agents        
ramsey_model(instruments=(R),planner_discount=beta,planner_discount_latex_name=$\beta$); 

//conduct stochastic simulations of the Ramsey problem
stoch_simul(order=1,irf=20,periods=500) pi_ann log_h R_ann log_C Z r_real;
evaluate_planner_objective;

