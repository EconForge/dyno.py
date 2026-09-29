function [modelIrf, error_indicator] = rbc_irf_matching_transformations(modelIrf, M_, options_mom_, ys_)
% -------------------------------------------------------------------------
% This file manipulates model IRFs to be consistent with empirical IRFS
% -------------------------------------------------------------------------
% INPUTS
% - modelIrf:        [options_mom_.irf by M_.endo_nbr by M_.exo_nbr]
%                                 array of IRFs for all model variables and all shocks
% - M_:              [structure]  Dynare model structure
% - options_mom_:    [structure]  Dynare options structure
% - ys_:             [double]     steady state values of all endogenous variables
% -------------------------------------------------------------------------
% OUTPUTS
% - modelIrf:        [options_mom_.irf by M_.endo_nbr by M_.exo_nbr]
%                                 modified array of IRFs for all model variables and all shocks
% - error_indicator: [boolean]    indicator of success (0) or failure (1)
% -------------------------------------------------------------------------
% This function is called by
% - mom.run
% -------------------------------------------------------------------------

% Copyright © 2024 Dynare Team
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

% initialize error indicator
error_indicator = 0;

% get indices of variables
idx_ghat = find(ismember(M_.endo_names,'ghat'));
idx_log_y = find(ismember(M_.endo_names,'log_y'));
idx_eps_g = find(ismember(M_.exo_names,'eps_g'));

% manipulate the model IRFs to match the empirical IRFs (e.g. cumsum, common scaling, trends, ratios, etc.)
modelIrf(:,idx_ghat,idx_eps_g)  = 100.*modelIrf(:,idx_ghat,idx_eps_g);
modelIrf(:,idx_log_y,idx_eps_g) = 100.*modelIrf(:,idx_log_y,idx_eps_g);

end