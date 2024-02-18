
/*
 * Example 1 from F. Collard (2001): "Stochastic simulations with DYNARE:
 * A practical guide" (see "guide.pdf" in the documentation directory).
 */

var a, b;
parameters a, b;
varexo e1, e2;

a = 12;
b = 4;

model;
p1 = 23 ; 
p2 = p1 + x;
a(1) = b + c; 
sin(a) = b(1);
end;

initval;
a = 12.0;
b = 0.8; // end of line comment
end;

stoch_simul;
