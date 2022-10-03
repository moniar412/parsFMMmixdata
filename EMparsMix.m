function [T, theta,plimix,lik,er]=EMparsMix(theta_init,T_init,tol,X,th_idx,G,P,Q,type,idxVar)
%
%   Author
%       Monia Ranalli
%       Department of Statistics
%       Sapienza University, Rome, Italy
%       Email : monia.ranalli@uniroma1.it
%
% OUTPUT__________________________________
% the function combines the E_stepMix and the M_stepMix until convergence
% INPUT___________________________________
% theta_init: a list containing the initial parameters f each group (by row)
% T_init: a initial classification matrix (it could be fuzzy or hard)
% tol: the level of tolerance for the convergence
% X: sample data
% th_idx: a vector keeping track to which variable the thresholds belongs to
% P: the number of variables
% G: the number of groups
% Q: the number of factors
% type: model type (1-8), that is CCC, CCU, CUC, CUU, UCC, UCU, UUC, UUU
% idxVar: 1 if ordinal variable, 0 if continuous
%
idxO=find(idxVar==1);
er=0;
O=length(idxO);
no.pairs=O*(O-1)/2;
pairs.index=combnk(1:O,2)';
if O>3 pairs.index=fliplr(pairs.index); end
it=0;
dev=Inf;
[theta,~]=M_parsMix(T_init,theta_init,X,G,th_idx,P,Q,type,idxVar,pairs.index);
T=T_init;
like=repmat(-10^10,1,size(pairs.index,2));
lik=-10^100;
while (dev>tol) ,
    it = it+1;
    theta0=theta;
    lik0=lik;
    [T,plimix]=E_stepMix(theta0,th_idx,P,G,X,idxVar,type,pairs.index);
    [theta,~]=M_parsMix(T,theta0,X,G,th_idx,P,Q,type,idxVar,pairs.index);
    lik=clik_obsMix(theta,th_idx,P,G,X,idxVar,type,pairs.index);
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

