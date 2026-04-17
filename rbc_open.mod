%var k i y a n c b r w;
var k i y a n c r w b rst;

varexo epsilon;

%parameters bet del alp nss khi eta rho rst;
parameters bet del alp nss khi eta rho;


bet=0.98;
alp=0.33;
del=0.025;
rho=0.95;
eta=1;
nss=0.33;
khi=(1-alp)*(1-nss)^eta/nss*(1/bet-1+del)/(1/bet-1+del-del*alp);
%rst=1/bet;


model;
1/c=bet*(r(1)+1-del)/c(1);
1/c=bet*rst/c(1);
w=khi*c/(1-n)^eta;
k=(1-del)*k(-1)+i;
y=a*k(-1)^alp*n^(1-alp);
log(a)=rho*log(a(-1))+epsilon;
w=(1-alp)*y/n;
r=alp*y/k(-1);
b=y-c-i+rst*b(-1);
rst=1/bet+exp(-b)-1;
end;


steady_state_model;
a=1;
r=1/bet-1+del;
n=nss;
k=(alp/r)^(1/(1-alp))*n;
y=k^alp*n^(1-alp);
w=(1-alp)*y/n;
i=del*k;
c=y-i;
b=0;
rst=1/bet;
end;


shocks;
var epsilon;stderr 0.009;
end;


resid;


check;


stoch_simul(irf=200);
%return
