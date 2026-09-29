// --+ options: json=compute, stochastic +--

var y x z v;

varexo ex ey ez ;

parameters a_y_1 a_y_2 b_y_1 b_y_2 b_x_1 b_x_2 d_y; // VAR parameters

parameters beta e_c_m c_z_1 c_z_2;               // PAC equation parameters

a_y_1 =  .2;
a_y_2 =  .3;
b_y_1 =  .1;
b_y_2 =  .4;
b_x_1 = -.1;
b_x_2 = -.2;
d_y = .5;


beta  =  .9;
e_c_m =  .1;
c_z_1 =  .7;
c_z_2 = -.3;

var_model(model_name=toto, eqtags=['eq:x', 'eq:y']);

pac_model(auxiliary_model_name=toto, discount=beta, model_name=pacman);

pac_target_info(pacman);
  target v;
  auxname_target_nonstationary vns;

  component y;
  auxname pv_y_;
  kind ll;

  component x;
  growth diff(x(-1));
  auxname pv_dx_;
  kind dd;

end;

model;

  [name='eq:y']
  y = a_y_1*y(-1) + a_y_2*diff(x(-1)) + b_y_1*y(-2) + b_y_2*diff(x(-2)) + ey ;


  [name='eq:x']
  diff(x) = b_x_1*y(-2) + b_x_2*diff(x(-1)) + ex ;

  [name='eq:v']
  v = x + d_y*y ;  // Composite target, no residuals here only variables defined in the auxiliary VAR model.

  [name='zpac']
  diff(z) = e_c_m*(pac_target_nonstationary(pacman)-z(-1)) + c_z_1*diff(z(-1))  + c_z_2*diff(z(-2)) + pac_expectation(pacman) + ez;

end;

shocks;
  var ex = .10;
  var ey = .15;
  var ez = .05;
end;

// Initialize the PAC model (build the Companion VAR representation for the auxiliary model).
pac.initialize('pacman');

// Update the parameters of the PAC expectation model (h0 and h1 vectors).
pac.update.expectation('pacman');

/*
**
** Simulate artificial dataset
**
*/

// Set initial conditions to zero.
initialconditions = dseries(zeros(10, M_.endo_nbr+M_.exo_nbr), 2000Q1, vertcat(M_.endo_names,M_.exo_names));

// Simulate the model for 5000 periods
TrueData = simul_backward_model(initialconditions, 5000);

/*
**
** Estimate PAC equation (using the artificial data)
**
*/


// Provide initial conditions for the estimated parameters
clear eparams
eparams.e_c_m   =  .9;
eparams.c_z_1   =  .5;
eparams.c_z_2   =  .2;

edata = TrueData;                // Set the dataset used for estimation
edata.ez = dseries(NaN, 2000Q1); // Remove residuals for the PAC equation from the database.

pac.estimate.nls('zpac', eparams, edata, 2005Q1:2005Q1+4000, 'fmincon'); // Should produce a table with the estimates (close to the calibration given in lines 21-23)
