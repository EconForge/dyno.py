%Définition des variables et des paramètres
var k i y a n c r w;
varexo epsilon;
parameters beta delta alpha nss khi eta rho;

%Valeur des paramètres
beta=0.985;
alpha=0.33;
delta=0.025;
rho=0.95;
eta=1;
nss=0.33;
khi=(1-alpha)*(1-nss)^eta/nss*(1/beta-1+delta)/(1/beta-1+delta-delta*alpha);

%Equations du modèle
model;
1/c=beta*(r(1)+1-delta)/c(1);
w=khi*c/(1-n)^eta;
k=(1-delta)*k(-1)+i;
y=a*k^alpha*n^(1-alpha);
log(a)=rho*log(a(-1))+epsilon;
w=(1-alpha)*y/n;
r=alpha*y/k(-1);
y=c+i;
end;

%Valeur des variables à l'état régulier
steady_state_model;
a=1;
r=1/beta+1-delta;
n=nss;
k=(alpha/r)^(1/(1-alpha))*n;
y=k^alpha*n^(1-alpha);
w=(1-alpha)*y/n;
i=delta*k;
c=y-i;
end;

%Définition des chocs exogènes
shocks;
var epsilon; 
stderr 0.009;
end;

steady;
check;

stoch_simul;