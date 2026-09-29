function log_prior_val=gali_2015_prior_restrictions(M_, dr, endo_steady_state, exo_steady_state, exo_det_steady_state, options_, dataset_, dataset_info);
% function prior_val=gali_2015_prior_restrictions(M_, dr, endo_steady_state, exo_steady_state, exo_det_steady_state, options_, dataset_, dataset_info);
% Example of a _prior_restrictions-file automatically called during
% estimation
% It imposes a prior of the slope of the New Keynesian Phillips Curve of
% 0.03. As the slope is a composite of other parameters with independent
% priors, a separate function is required to do this.

% Copyright © 2021-2023 Dynare Team
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
NumberOfParameters = M_.param_nbr;
for ii = 1:NumberOfParameters
  paramname = M_.param_names{ii};
  eval([ paramname ' = M_.params(' int2str(ii) ');']);
end

Omega=(1-alppha)/(1-alppha+alppha*epsilon);
lambda=(1-theta)*(1-betta*theta)/theta*Omega;      %defined on page 61
kappa=lambda*(siggma+(varphi+alppha)/(1-alppha));     %defined on page 63

prior_mean=0.03;
prior_std=0.02;
log_prior_val=log(normpdf(kappa,prior_mean,prior_std));
