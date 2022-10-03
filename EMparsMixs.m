function [T, theta,plimix,lik,er]=EMparsMixs(theta_init,T_init,tol,X,th_idx,G,P,Q,type,idxVar)
%
%   Author
%       Monia Ranalli
%       Department of Statistics
%       Sapienza University, Rome, Italy
%       Email : monia.ranalli@uniroma1.it
%
% OUTPUT__________________________________
% the function combines the E_stepMixs and the M_stepMixs until convergence
% INPUT___________________________________
% theta_init: a list containing the initial parameters f each group (by row)
% T_init: a initial classification matrix (it could be fuzzy or hard)
% tol: the level of tolerance for the convergence
% X: sample data
% th_idx: a vector keeping track to which variable the thresholds belongs to
% P: the number of variables
% G: the number of groups
% Q: the number of factors
% type: model type (9-12), that is SC)CC, (SC)CU, (SC)UC, (SC)UU
% idxVar: 1 if ordinal variable, 0 if continuous
%
idxC=find(idxVar==0);
idxO=find(idxVar==1);
er=0;
O=length(idxO);
no.pairs=O*(O-1)/2;
pairs.index=combnk(1:O,2)';
if O>3 pairs.index=fliplr(pairs.index); end
it=0;
dev=Inf;
[theta,d]=M_parsMixs(T_init,theta_init,X,G,th_idx,P,Q,type,idxVar,pairs.index);
T=T_init;
lik=-10^100;
while (dev>tol),
    it = it+1;
    theta0=theta;
    lik0=lik;
    [T,plimix]=E_stepMixs(theta0,th_idx,P,G,X,idxVar,type,pairs.index);
    [theta,d]=M_parsMixs(T,theta0,X,G,th_idx,P,Q,type,idxVar,pairs.index);
    lik=clik_obsMixs(theta,th_idx,P,G,X,idxVar,type,pairs.index);
    dev=(lik-lik0)/abs(lik0);
    if dev<-0.000001
        disp('------------------------------------------- error -------------------------------------')
        er=dev;
    end
    if it>50,
        dev=-50;
    end
    
    disp(sprintf('mixture(%g): clik=%g, iter=%g, dev=%g',G,lik,it,dev))
    
    
end



end

