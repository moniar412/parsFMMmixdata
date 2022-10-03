function [M,d]=M_parsMix(T,theta,X,G,th_idx,P,Q,type,idxVar,pairsindex)
%
%   Author
%       Monia Ranalli
%       Department of Statistics
%       Sapienza University, Rome, Italy
%       Email : monia.ranalli@uniroma1.it
%
% OUTPUT__________________________________
% compute the M-step of the EM like algorithm for models 1-8; Since there is not a closed form for the parameter estimates, an iterative optimization
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
% type: model type (1-8), that is CCC, CCU, CUC, CUU, UCC, UCU, UUC, UUU
% idxVar: 1 if ordinal variable, 0 if continuous
% pairsindex: indexes for the pairs of variables
%
idxO=find(idxVar==1);
pairs.index=pairsindex;
O=length(idxO);
no.pairs=O*(O-1)/2;
j=find(theta.gamma>1);
if (type<5)
    V=reshape(1:(P*Q),P,Q);
    Vu=(triu(V(1:Q,1:Q),1));
    idL0=(Vu(Vu>0))';
    idLall=1:(P*Q);
    %if 0 estimate the parameters, if 1 parameter is 0
    idLlog=ismember(idLall,idL0);
    idDiag=diag(V(1:Q,1:Q))';
    idLD=ismember(idLall,idDiag);
else
    V=reshape(1:(P*Q*G),P,Q,G);
    idL0=[];
    idDiag=[];
    for g=1:G,
        Vu=(triu(V(1:Q,1:Q,g),1));
        idDiag=[idDiag,diag(V(1:Q,1:Q,g))'];
        idL0=[idL0,(Vu(Vu>0))'];
    end
    idLall=1:(P*Q*G);
    %if 0 estimate the parameters, if 1 parameter is 0
    idLlog=ismember(idLall,idL0);
    idLD=ismember(idLall,idDiag);
end
Lvec=reshape(theta.L,1,[]);
param0=[reshape(theta.mu',1,[]),theta.gamma(theta.gamma>1),Lvec(idLlog==0), reshape(theta.psi,1,[]) ];
d=length(param0);
Llen=length(Lvec(idLlog==0));
Plen=length(reshape(theta.psi,1,[]));
options=optimset('fmincon');
options = optimset(options,'Algorithm','interior-point','LargeScale','off','Display','off','UseParallel',true,'MaxFunEvals',30*(length(param0)),'MaxIter',10*(length(param0)));
cl=repmat(-1,1,length(idLlog));
cl(idLD==1)=0;
cl=cl(idLlog==0);
lb=[repmat(-6,1,G*P)    repmat(1,1,length(th_idx)-2*O),cl,repmat(0.05,1,Plen)];
ub=[repmat(+6,1,G*P)   repmat(+10,1,length(th_idx)-2*O),repmat(1,1,Llen),repmat(1,1,Plen)];
[par,~]=fmincon(@(param0)myfunAllModsMix1(param0,th_idx,X,T,P,G,Q,j,type,idxVar,pairs.index,idLlog),param0,[],[],[],[],lb,ub,[],options);
mu=(reshape(par(1:P*G),P,G))';
b=par((P*G+1):(P*G+length(th_idx)-2*O));
a=1:length(th_idx);
a(j)=[];  bb=repmat([0 1],1,O);
br=zeros(1,length(th_idx)); br(j)=b; br(a)=bb;
Sig=ones(P,P,G);
if (type==1),
    L0=par((P*G+length(th_idx)-2*O+1):(P*G+length(th_idx)-2*O+P*Q-Q*(Q-1)/2));
    L00=zeros(1,P*Q);
    L00(idLlog==0)=L0;
    L=reshape(L00,P,Q);
    psi=par((P*G+length(th_idx)-2*O+P*Q-Q*(Q-1)/2+1):(P*G+length(th_idx)-2*O+P*Q-Q*(Q-1)/2+1));
    J=L*L'+diag(repmat(psi,[P,1]));
    [ei, eig1]=eig(J);
    eig1(eig1<0)=0.001;
    Sig=ei*eig1*ei';
    Sig=repmat(Sig,[1,1,G]);
elseif (type==2),
    L0=par((P*G+length(th_idx)-2*O+1):(P*G+length(th_idx)-2*O+P*Q-Q*(Q-1)/2));
    L00=zeros(1,P*Q);
    L00(idLlog==0)=L0;
    L=reshape(L00,P,Q);
    psi=par((P*G+length(th_idx)-2*O+P*Q-Q*(Q-1)/2+1):(P*G+length(th_idx)-2*O+P*Q-Q*(Q-1)/2+P));
    J=L*L'+diag(psi);
    [ei eig1]=eig(J);
    eig1(eig1<0)=0.001;
    Sig=ei*eig1*ei';
    Sig=repmat(Sig,[1,1,G]);
elseif (type==3),
    L0=par((P*G+length(th_idx)-2*O+1):(P*G+length(th_idx)-2*O+P*Q-Q*(Q-1)/2));
    L00=zeros(1,P*Q);
    L00(idLlog==0)=L0;
    L=reshape(L00,P,Q);
    psi=par((P*G+length(th_idx)-2*O+P*Q-Q*(Q-1)/2+1):(P*G+length(th_idx)-2*O+P*Q-Q*(Q-1)/2+G));
    for g=1:G,
        J=L*L'+diag(repmat(psi(g),[P,1]));
        [ei eig1]=eig(J);
        eig1(eig1<0)=0.001;
        Sig(:,:,g)= ei*eig1*ei';
    end
elseif (type==4),
    L0=par((P*G+length(th_idx)-2*O+1):(P*G+length(th_idx)-2*O+P*Q-Q*(Q-1)/2));
    L00=zeros(1,P*Q);
    L00(idLlog==0)=L0;
    L=reshape(L00,P,Q);
    psi=reshape(par((P*G+length(th_idx)-2*O+P*Q-Q*(Q-1)/2+1):(P*G+length(th_idx)-2*O+P*Q-Q*(Q-1)/2+P*G)),P,G);
    for g=1:G,
        J=L*L'+diag(psi(:,g));
        [ei eig1]=eig(J);
        eig1(eig1<0)=0.001;
        Sig(:,:,g)= ei*eig1*ei';
    end
elseif (type==5),
    L0=par((P*G+length(th_idx)-2*O+1):(P*G+length(th_idx)-2*O+P*Q*G-G*Q*(Q-1)/2));
    L00=zeros(1,P*Q*G);
    L00(idLlog==0)=L0;
    L=reshape(L00,P,Q,G);
    psi=par((P*G+length(th_idx)-2*O+P*Q*G-G*Q*(Q-1)/2+1):(P*G+length(th_idx)-2*O+P*Q*G-G*Q*(Q-1)/2+1));
    for g=1:G,
        J=L(:,:,g)*L(:,:,g)'+diag(repmat(psi,[P,1]));
        [ei eig1]=eig(J);
        eig1(eig1<0)=0.001;
        Sig(:,:,g)= ei*eig1*ei';
    end
elseif (type==6),
    L0=par((P*G+length(th_idx)-2*O+1):(P*G+length(th_idx)-2*O+P*Q*G-G*Q*(Q-1)/2));
    L00=zeros(1,P*Q*G);
    L00(idLlog==0)=L0;
    L=reshape(L00,P,Q,G);
    psi=par((P*G+length(th_idx)-2*O+P*Q*G-G*Q*(Q-1)/2+1):(P*G+length(th_idx)-2*O+P*Q*G-G*Q*(Q-1)/2+P));
    for g=1:G,
        J=L(:,:,g)*L(:,:,g)'+diag(psi);
        [ei eig1]=eig(J);
        eig1(eig1<0)=0.001;
        Sig(:,:,g)= ei*eig1*ei';
    end
elseif (type==7),
    L0=par((P*G+length(th_idx)-2*O+1):(P*G+length(th_idx)-2*O+P*Q*G-G*Q*(Q-1)/2));
    L00=zeros(1,P*Q*G);
    L00(idLlog==0)=L0;
    L=reshape(L00,P,Q,G);
    psi=par((P*G+length(th_idx)-2*O+P*Q*G-G*Q*(Q-1)/2+1):(P*G+length(th_idx)-2*O+P*Q*G-G*Q*(Q-1)/2+G));
    for g=1:G,
        J=L(:,:,g)*L(:,:,g)'+diag(repmat(psi(g),[P,1]));
        [ei eig1]=eig(J);
        eig1(eig1<0)=0.001;
        Sig(:,:,g)= ei*eig1*ei';
    end
elseif (type==8),
    L0=par((P*G+length(th_idx)-2*O+1):(P*G+length(th_idx)-2*O+P*Q*G-G*Q*(Q-1)/2));
    L00=zeros(1,P*Q*G);
    L00(idLlog==0)=L0;
    L=reshape(L00,P,Q,G);
    psi=reshape(par((P*G+length(th_idx)-2*O+P*Q*G-G*Q*(Q-1)/2+1):(P*G+length(th_idx)-2*O+P*Q*G-G*Q*(Q-1)/2+P*G)),P,G);
    for g=1:G,
        J=L(:,:,g)*L(:,:,g)'+diag(psi(:,g));
        [ei eig1]=eig(J);
        eig1(eig1<0)=0.001;
        Sig(:,:,g)=ei*eig1*ei';
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
M=struct('pg',pg,'mu',mu, 'sigma',sigma2,'L',L,'psi',psi,'cors',cors,'gamma',br);
end


