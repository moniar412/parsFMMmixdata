function [M,d]=M_parsMixs(T,theta,X,G,th_idx,P,Q,type,idxVar,pairsindex)
%
%   Author
%       Monia Ranalli
%       Department of Statistics
%       Sapienza University, Rome, Italy
%       Email : monia.ranalli@uniroma1.it
%
% OUTPUT__________________________________
% compute the M-step of the EM like algorithm for models 9-12; Since there is not a closed form for the parameter estimates, an iterative optimization
% is used (fminsearch). By default fminsearch performs minimization, but it maximize because the function is
% multiplied by -1. CAUTION: results may change if 'MaxFunEvals' and
% 'MaxIter' are changed. These parameters may be tuned according to the
% maximization setting.
% INPUT___________________________________
% T: classification matrix
% theta: a list containing the parameters of each group (by row)
% X: sample data
% G: the number of groups
% th_idx: a vector keeping track to which variable the thresholds belongs to
% P: the number of ordinal variables
% Q: the number of factors
% type: model type (9-12), that is SC)CC, (SC)CU, (SC)UC, (SC)UU
% idxVar: 1 if ordinal variable, 0 if continuous
% pairsindex: indexes for the pairs of variables
%
idxO=find(idxVar==1);
pairs.index=pairsindex;
O=length(idxO);
no.pairs=O*(O-1)/2;
j=find(theta.gamma>1);
Lvec=reshape(theta.L,1,[]);
l0=reshape(theta.l0(2:end,:),1,[]);
param0=[reshape(theta.mu',1,[]),theta.gamma(theta.gamma>1),Lvec,l0, reshape(theta.psi,1,[]) ];
d=length(param0);
Plen=length(reshape(theta.psi,1,[]));
options=optimset('fmincon');
options = optimset(options,'Algorithm','interior-point','LargeScale','off','Display','off','UseParallel',true,'MaxFunEvals',20*(length(param0)),'MaxIter',5*(length(param0)));
lb=[repmat(-6,1,G*P)    repmat(1,1,length(th_idx)-2*O),repmat(-1,1,P*Q),repmat(0,1,Q*(G-1)),repmat(0.05,1,Plen)];
ub=[repmat(+6,1,G*P)   repmat(+10,1,length(th_idx)-2*O),repmat(1,1,P*Q),repmat(2,1,Q*(G-1)),repmat(1,1,Plen)];
[par,f]=fmincon(@(param0)myfunAllModsMix1s(param0,th_idx,X,T,P,G,Q,j,type,idxVar,pairs.index),param0,[],[],[],[],lb,ub,[],options);
mu=(reshape(par(1:P*G),P,G))';
b=par((P*G+1):(P*G+length(th_idx)-2*O));
a=1:length(th_idx);
a(j)=[];  bb=repmat([0 1],1,O);
br=zeros(1,length(th_idx)); br(j)=b; br(a)=bb;
Sig=ones(P,P,G);
if (type==9),
    L0=par((P*G+length(th_idx)-2*O+1):(P*G+length(th_idx)-2*O+P*Q));
    L=reshape(L0,P,Q);
    l0=par((P*G+length(th_idx)-2*O+P*Q+1):(P*G+length(th_idx)-2*O+P*Q+(G-1)*Q));
    l0=reshape(l0,G-1,Q);
    l0=[ones(1,Q);l0];
    psi=par((P*G+length(th_idx)-2*O+P*Q+(G-1)*Q+1):(P*G+length(th_idx)-2*O+P*Q+(G-1)*Q+1));
    for g=1:G,
        J=L*diag(l0(g,:))*L'+diag(repmat(psi,[P,1]));
        [ei, eig1]=eig(J);
        eig1(eig1<0)=0.001;
        Sig(:,:,g)=ei*eig1*ei';
    end
elseif (type==10),
    L0=par((P*G+length(th_idx)-2*O+1):(P*G+length(th_idx)-2*O+P*Q));
    L=reshape(L0,P,Q);
    l0=par((P*G+length(th_idx)-2*O+P*Q+1):(P*G+length(th_idx)-2*O+P*Q+(G-1)*Q));
    l0=reshape(l0,G-1,Q);
    l0=[ones(1,Q);l0];
    psi=par((P*G+length(th_idx)-2*O+P*Q+(G-1)*Q+1):(P*G+length(th_idx)-2*O+P*Q+(G-1)*Q+P));
    for g=1:G,
        J=L*diag(l0(g,:))*L'+diag(psi);
        [ei, eig1]=eig(J);
        eig1(eig1<0)=0.001;
        Sig(:,:,g)=ei*eig1*ei';
    end
elseif (type==11),
    L0=par((P*G+length(th_idx)-2*O+1):(P*G+length(th_idx)-2*O+P*Q));
    L=reshape(L0,P,Q);
    l0=par((P*G+length(th_idx)-2*O+P*Q+1):(P*G+length(th_idx)-2*O+P*Q+(G-1)*Q));
    l0=reshape(l0,G-1,Q);
    l0=[ones(1,Q);l0];
    psi=par((P*G+length(th_idx)-2*O+P*Q+(G-1)*Q+1):(P*G+length(th_idx)-2*O+P*Q+(G-1)*Q+G));
    for g=1:G,
        J=L*diag(l0(g,:))*L'+diag(repmat(psi(g),[P,1]));
        [ei, eig1]=eig(J);
        eig1(eig1<0)=0.001;
        Sig(:,:,g)= ei*eig1*ei';
    end
elseif (type==12),
    L0=par((P*G+length(th_idx)-2*O+1):(P*G+length(th_idx)-2*O+P*Q));
    L=reshape(L0,P,Q);
    l0=par((P*G+length(th_idx)-2*O+P*Q+1):(P*G+length(th_idx)-2*O+P*Q+(G-1)*Q));
    l0=reshape(l0,G-1,Q);
    l0=[ones(1,Q);l0];
    psi=reshape(par((P*G+length(th_idx)-2*O+P*Q+(G-1)*Q+1):(P*G+length(th_idx)-2*O+P*Q+(G-1)*Q+P*G)),P,G);
    for g=1:G,
        J=L*diag(l0(g,:))*L'+diag(psi(:,g));
        [ei, eig1]=eig(J);
        eig1(eig1<0)=0.001;
        Sig(:,:,g)= ei*eig1*ei';
    end
end
for i=1:P,
    br(th_idx==i)=sort(br(th_idx==i));
end
sigma2=ones(G,P);
cors=zeros(G,P*(P-1)/2);
r=(find(tril(ones(size(ones(P,P))),-1)));
for g=1:G,
    sigma2(g,:)=diag(Sig(:,:,g))';
    c=tril(diag(sqrt(diag(Sig(:,:,g))).^-1)*Sig(:,:,g)*diag(sqrt(diag(Sig(:,:,g))).^-1),-1);
    cors(g,:)=c(r)';
end
N=size(X,1);
pg=mean(sum(T,1)/N,3);
M=struct('pg',pg,'mu',mu, 'sigma',sigma2,'L',L,'l0',l0,'psi',psi,'cors',cors,'gamma',br);
end


