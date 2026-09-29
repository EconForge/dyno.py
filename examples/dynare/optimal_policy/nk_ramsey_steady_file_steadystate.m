function [ys,params,check] = nk_ramsey_steady_file_steadystate(ys,exo,M_,options_)
% function [ys,params,check] = nk_ramsey_steady_file_steadystate(ys,exo,M_,options_)
% computes the steady state for the nk_ramsey_steady_file.mod, conditional on
% the instrument value provided
%
% Inputs:
%   - ys        [vector] vector of initial values for the steady state of
%                   the endogenous variables
%   - exo       [vector] vector of values for the exogenous variables
%   - M_        [structure] Dynare model structure
%   - options   [structure] Dynare options structure
%
% Output:
%   - ys        [vector] vector of steady state values for the the endogenous variables
%   - params    [vector] vector of parameter values
%   - check     [scalar] set to 0 if steady state computation worked and to
%                    1 of not (allows to impose restrictions on parameters)

% Copyright © 2020-2022 Dynare Team
%
% This file is part of Dynare.
%
% Dynare is free software: you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation, either version 3 of the License, or
% (at your option) any later version.
%
% Dynare is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
% GNU General Public License for more details.
%
% You should have received a copy of the GNU General Public License
% along with Dynare.  If not, see <https://www.gnu.org/licenses/>.

% read out parameters to access them with their name
beta=NaN; %make parameter known to Matlab function, prevents crashes due to Matlab function with same name;
          %will be overwritten next

NumberOfParameters = M_.param_nbr;
for ii = 1:NumberOfParameters
  paramname = M_.param_names{ii};
  eval([ paramname ' = M_.params(' int2str(ii) ');']);
end

% read in instrument values
for ii = 1:size(options_.instruments,1)
  eval([options_.instruments{ii} ' = ys(strmatch(options_.instruments{ii},M_.endo_names,''exact'')) ;']);
end
% initialize indicator
check = 0;


%% Enter model equations here

    Z=0;
    pi=(R+1)*beta;
    C=sqrt((1+1/theta*((1-beta)*(pi-1)*pi-(tau-1/(theta-1))*(1-theta)))/(chi*(1+phi/2*(pi-1)^2)));
    h=C*(1+phi/2*(pi-1)^2);
    log_C=log(C);
    log_h=log(h);
    pi_ann=4*log(pi);
    R_ann=4*R;
    r_real=4*log((1+R)/pi);
    y_nat=sqrt((theta-1)/theta*(1+tau)/chi);
    y_gap=log_C-log(y_nat);

%% end own model equations

params=NaN(NumberOfParameters,1);
for iter = 1:length(M_.params) %update parameters set in the file
  eval([ 'params(' num2str(iter) ') = ' M_.param_names{iter} ';' ])
end

NumberOfEndogenousVariables = M_.orig_endo_nbr; %auxiliary variables are set automatically
for ii = 1:NumberOfEndogenousVariables
  varname = M_.endo_names{ii};
  eval(['ys(' int2str(ii) ') = ' varname ';']);
end
