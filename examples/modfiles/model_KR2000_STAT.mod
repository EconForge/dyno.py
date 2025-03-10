%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Canonical Real Business Cycle (RBC) model based on King and Rebelo (2000), Handbook of Macroeconomics %
% Macroeconomie 2: Fluctuations ECO_4MA05_AE / Franck Malherbet                                         %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%
% Environment
%

%
% Endogeneous variables
%
var 
 y              ${y}$                   (long_name='Output')
 c              ${c}$                   (long_name='Consumpation')
 i              ${i}$                   (long_name='Investment')
 k              ${k}$                   (long_name='Capital')
 H              ${H}$                   (long_name='Hours worked')
 r              ${r}$                   (long_name='Real interest rate')
 w              ${w}$                   (long_name='Real wage')
 A              ${A}$                   (long_name='Total factor productivity')
 p              ${p}$                   (long_name='Labor productivity')
 expr
;
      
// dyno doesn't support predetermined variables yet
// predetermined_variables k;

varexo epsilon;                                                             % Shock(s)

%
% Parametrization/Calibration (see King and Rebelo, 2000)
% 
parameters 
 alpha          $\alpha$                (long_name='capital share') 
 beta           $\beta$                 (long_name='discount factor')
 delta          $\delta$                (long_name='depreciation rate') 
 gamma          $\gamma$                (long_name='growth rate of technical progress')
 phi            $\phi$                  (long_name='labor disutility')
 rho            $\rho$                  (long_name='TFP shock autocorrelation')
 sigmae         $\sigma_{\epsilon}$     (long_name='TFP shock volatility')    
;

alpha           = 1/3;
beta            = 0.984;
delta           = 0.025;
gamma           = 1.004;
phi             = 3.48;
rho             = 0.974;
sigmae          = 0.0072;   

%
% Model (FOCs, steady-state)
%

model;
gamma       = beta*c/c(+1)*(1+r(+1));
H           = 1-phi*c/w;
y           = A*k(-1)^alpha*H^(1-alpha);
r           = alpha*y/k(-1)-delta;
w           = (1-alpha)*y/H;
i           = gamma*k(0)-(1-delta)*k(-1);
y           = c+i;
p           = y/H;
log(A)      = rho*log(A(-1))+epsilon;
expr        = exp(r);
end;

steady_state_model;
A           = 1;
r           = gamma/beta-1;
k_H         = (alpha/(r+delta))^(1/(1-alpha));
y_H         = k_H^alpha;
w           = (1-alpha)*y_H;
c_H         = y_H-(gamma-1+delta)*k_H;
H           = 1/(1+phi*c_H/w);
k           = k_H*H;
c           = c_H*H;
y           = y_H*H;
i           = y-c;
p           = y/H;
expr        = exp(r);
end;

steady;

check;

shocks;
var epsilon;
stderr sigmae; 
end;

stoch_simul(periods=5000, drop=200, hp_filter=1600, order=1, irf=0);
% EOF