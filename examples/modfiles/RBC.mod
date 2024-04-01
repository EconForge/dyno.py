var c, r, w, n, k, i, y, a;
varexo epsilon;
parameters beta, delta, khi, eta, alpha, rho, nss;

beta=0.985;
delta=0.025;
nss=0.33;
eta=1.0;
alpha=.33;
rho=.95;
khi=(1-alpha)*(1-nss)^eta/nss*(1/beta-1+delta)/(1/beta-1+delta-delta*alpha);

model;
1/c = beta*(r(1)+1-delta)/c(1);
w = khi*c/(1-n)^eta;
k = (1-delta)*k(-1)+i;
y = a*k(-1)^alpha*n^(1-alpha);
log(a) = rho*log(a(-1))+epsilon;
w = (1-alpha)*y/n;
r = alpha*y/k(-1);
y = c+i;
end;

steady_state_model;
a = 1;
r = 1/beta-1+delta;
n = nss;
k = (alpha/r)^(1/(1-alpha))*n;
y = k^alpha*n^(1-alpha);
w = (1-alpha)*y/n;
i = delta*k;
c = y-i;
end;

shocks;
var epsilon; stderr .009;
end;