function f=myfunAllModsMix1s(param,th_idx,X,T,P,G,Q,j,type,idxVar,pairsindex)
%
%   Author
%       Monia Ranalli
%       Department of Statistics
%       Sapienza University, Rome, Italy
%       Email : monia.ranalli@uniroma1.it
%
% OUTPUT__________________________________
% maximized complete data composite log-likelihood for models 9-12
% INPUT___________________________________
% param: a vector containing the parameters to estimate
% th_idx: the vector keeping track to which variable the thresholds belongs to
% X: sample data
% T: classification matrix
% P: the number of variables
% G: the number of groups
% Q: the number of factors
% j: indexes for varying thresholds (each variables has the first two thresholds set to 0 and 1, respectively)
% type: model type (9-12), that is SC)CC, (SC)CU, (SC)UC, (SC)UU
% idxVar: 1 if ordinal variable, 0 if continuous
% pairsindex: indexes for the pairs of variables
%
idxC=find(idxVar==0);
idxO=find(idxVar==1);
N=size(X,1);
pairs.index=pairsindex;
O=length(idxO);
mu=(reshape(param(1:P*G),P,G))';
b=param((P*G+1):(P*G+length(th_idx)-2*O));
a=1:length(th_idx);
a(j)=[];  bb=repmat([0 1],1,O);
br=zeros(1,length(th_idx)); br(j)=b; br(a)=bb;
Sig=ones(P,P,G);
if (type==9),
    L0=param((P*G+length(th_idx)-2*O+1):(P*G+length(th_idx)-2*O+P*Q));
    L=reshape(L0,P,Q);
    l0=param((P*G+length(th_idx)-2*O+P*Q+1):(P*G+length(th_idx)-2*O+P*Q+(G-1)*Q));
    l0=reshape(l0,G-1,Q);
    l0=[ones(1,Q);l0];
    psi=param((P*G+length(th_idx)-2*O+P*Q+(G-1)*Q+1):(P*G+length(th_idx)-2*O+P*Q+(G-1)*Q+1));
    for g=1:G,
        J=L*diag(l0(g,:))*L'+diag(repmat(psi,[P,1]));
        [ei, eig1]=eig(J);
        eig1(eig1<0)=0.001;
        Sig(:,:,g)=ei*eig1*ei';
    end
elseif (type==10),
    L0=param((P*G+length(th_idx)-2*O+1):(P*G+length(th_idx)-2*O+P*Q));
    L=reshape(L0,P,Q);
    l0=param((P*G+length(th_idx)-2*O+P*Q+1):(P*G+length(th_idx)-2*O+P*Q+(G-1)*Q));
    l0=reshape(l0,G-1,Q);
    l0=[ones(1,Q);l0];
    psi=param((P*G+length(th_idx)-2*O+P*Q+(G-1)*Q+1):(P*G+length(th_idx)-2*O+P*Q+(G-1)*Q+P));
    for g=1:G,
        J=L*diag(l0(g,:))*L'+diag(psi);
        [ei, eig1]=eig(J);
        eig1(eig1<0)=0.001;
        Sig(:,:,g)=ei*eig1*ei';
    end
elseif (type==11),
    L0=param((P*G+length(th_idx)-2*O+1):(P*G+length(th_idx)-2*O+P*Q));
    L=reshape(L0,P,Q);
    l0=param((P*G+length(th_idx)-2*O+P*Q+1):(P*G+length(th_idx)-2*O+P*Q+(G-1)*Q));
    l0=reshape(l0,G-1,Q);
    l0=[ones(1,Q);l0];
    psi=param((P*G+length(th_idx)-2*O+P*Q+(G-1)*Q+1):(P*G+length(th_idx)-2*O+P*Q+(G-1)*Q+G));
    for g=1:G,
        J=L*diag(l0(g,:))*L'+diag(repmat(psi(g),[P,1]));
        [ei, eig1]=eig(J);
        eig1(eig1<0)=0.001;
        Sig(:,:,g)= ei*eig1*ei';
    end
elseif (type==12),
    L0=param((P*G+length(th_idx)-2*O+1):(P*G+length(th_idx)-2*O+P*Q));
    L=reshape(L0,P,Q);
    l0=param((P*G+length(th_idx)-2*O+P*Q+1):(P*G+length(th_idx)-2*O+P*Q+(G-1)*Q));
    l0=reshape(l0,G-1,Q);
    l0=[ones(1,Q);l0];
    psi=reshape(param((P*G+length(th_idx)-2*O+P*Q+(G-1)*Q+1):(P*G+length(th_idx)-2*O+P*Q+(G-1)*Q+P*G)),P,G);
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
cors=zeros(G,P*(P-1)/2);
sigma2=ones(G,P);
r=(find(tril(ones(size(ones(P,P))),-1)));
for g=1:G,
    sigma2(g,:)=diag(Sig(:,:,g))';
    c=tril(diag(sqrt(diag(Sig(:,:,g))).^-1)*Sig(:,:,g)*diag(sqrt(diag(Sig(:,:,g))).^-1),-1);
    cors(g,:)=c(r)';
end
Xo=X(:,idxO);
Xc=X(:,idxC);
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
for idx=1:size(pairs.index,2),
    for g=1:G,
        s2c1=Sig(idxC,idxC,g);
        S2=s2c1+0.01*eye(length(idxC));
        Sigr=Sig(idxO(pairs.index(:,idx)),idxC,g)/S2;
        s2c=Sig(idxO(pairs.index(:,idx)),idxO(pairs.index(:,idx)),g)-Sigr*Sig(idxC,idxO(pairs.index(:,idx)),g);
        s2c=s2c+0.01*eye(2);
        [~,r2c]=cov2corr(s2c);
        r2c=real(r2c);
        muc=mu0(g,idxO(pairs.index(:,idx)))'+Sig(idxC,idxO(pairs.index(:,idx)),g)'/S2*(Xc-m(g,:))';
        low=(t11(:,idxO(pairs.index(:,idx)),g)-muc')./repmat(sqrt(diag(s2c)'),size(t11,1),1);
        up=(t22(:,idxO(pairs.index(:,idx)),g)-muc')./repmat(sqrt(diag(s2c)'),size(t11,1),1);
        B=[low,up];
        t3=bvnP(real(B(:,1)),real(B(:,3)),real(B(:,2)),real(B(:,4)),r2c(1,2));
        t4=mvnpdf(Xc,real(m(g,:)),real(S2));
        plimix(:,g,idx)=t3.*t4;
        
    end
end
f=log(plimix);
plik=f.*T;
f=sum(reshape((plik),1,numel(plik)));
f=-f;
end
