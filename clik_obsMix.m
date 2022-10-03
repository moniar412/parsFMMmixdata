function cllik=clik_obsMix(theta,th_idx,P,G,X,idxVar,type,pairsindex)
%
%   Author
%       Monia Ranalli
%       Department of Statistics
%       Sapienza University, Rome, Italy
%       Email : monia.ranalli@uniroma1.it
% OUTPUT__________________________________
% the observed composite log-likelihood
% INPUT___________________________________
% theta:  a list containing the parameters of each group (by row)
% th_idx: a vector keeping track to which variable the thresholds belongs to
% P: the number of variables
% G: the number of groups
% X: sample data
% idxVar: 1 if ordinal variable, 0 if continuous
% type: model type (1-8), that is CCC, CCU, CUC, CUU, UCC, UCU, UUC, UUU
% pairsindex: indexes for the pairs of variables
idxC=find(idxVar==0);
idxO=find(idxVar==1);
Xc=X(:,idxC);
N=size(X,1);
O=length(idxO);
pairs.index=pairsindex;
L=theta.L;
psi=theta.psi;
Sig=ones(P,P,G);
br=theta.gamma;
mu=theta.mu;
Xo=X(:,idxO);
if (type==1),
    Sig=L*L'+diag(repmat(psi,[P,1]));
    Sig=repmat(Sig,[1,1,G]);
elseif (type==2),
    Sig=L*L'+diag(psi);
    Sig=repmat(Sig,[1,1,G]);
elseif (type==3),
    for g=1:G,
        Sig(:,:,g)=L*L'+diag(repmat(psi(g),[P,1]));
    end
elseif (type==4),
    for g=1:G,
        Sig(:,:,g)=L*L'+diag(psi(:,g));
    end
elseif (type==5),
    for g=1:G,
        Sig(:,:,g)=L(:,:,g)*L(:,:,g)'+diag(repmat(psi,[P,1]));
    end
elseif (type==6),
    for g=1:G,
        Sig(:,:,g)=L(:,:,g)*L(:,:,g)'+diag(psi);
    end
elseif (type==7),
    for g=1:G,
        Sig(:,:,g)=L(:,:,g)*L(:,:,g)'+diag(repmat(psi(g),[P,1]));
    end
elseif (type==8),
    for g=1:G,
        Sig(:,:,g)=L(:,:,g)*L(:,:,g)'+diag(psi(:,g));
    end
end
plimix=ones(N,G,size(pairs.index,2));
br1=repmat(br,G,1);
t1=ones(N,size(Xo,2),G);
t2=ones(N,size(Xo,2),G);
for g=1:G,
    [~,gamma1_Pext1,gamma1_Pext2]=vargamma(br1(g,:),th_idx,O);
    for p=1:size(Xo,2),
        val=Xo(:,p);
        t1(:,p,g)=gamma1_Pext1{p}(val);
        t2(:,p,g)=gamma1_Pext2{p}(val);
    end
end
t11=zeros(N,P,G);
t22=zeros(N,P,G);
t11(:,idxO,:)=t1;
t22(:,idxO,:)=t2;
m=mu(:,idxVar==0,:);
mu0(:,idxO)=mu(:,idxVar==1,:);
parfor idx=1:size(pairs.index,2),
    for g=1:G,
        s2c1=Sig(idxC,idxC,g);
        S2=s2c1+0.001*eye(length(idxC));
        Sigr=Sig(idxO(pairs.index(:,idx)),idxC,g)/S2;
        s2c=Sig(idxO(pairs.index(:,idx)),idxO(pairs.index(:,idx)),g)-Sigr*Sig(idxC,idxO(pairs.index(:,idx)),g);
        s2c=s2c+0.001*eye(2);
        [~,r2c]=cov2corr(s2c);
        muc=mu0(g,idxO(pairs.index(:,idx)))'+Sig(idxC,idxO(pairs.index(:,idx)),g)'/S2*(Xc-m(g,:))';
        low=(t11(:,idxO(pairs.index(:,idx)),g)-muc')./repmat(sqrt(diag(s2c)'),size(t11,1),1);
        up=(t22(:,idxO(pairs.index(:,idx)),g)-muc')./repmat(sqrt(diag(s2c)'),size(t11,1),1);
        B=[low,up];
        t3=bvnP(B(:,1),B(:,3),B(:,2),B(:,4),r2c(1,2));
        t4=mvnpdf(Xc,m(g,:),S2);
        plimix(:,g,idx)=t3.*t4;
        
    end
end
lik1=log(plimix);
lik1=sum(lik1,3);
ind=(lik1<-700);
U=exp(-700*ind+lik1.*(1-ind))*diag(theta.pg);
cllik=sum(log(sum(U,2)));
end
