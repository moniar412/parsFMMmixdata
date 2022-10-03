function f=myfunAllModsMix1(param,th_idx,X,T,P,G,Q,j,type,idxVar,pairsindex,idLlog)
%
%   Author
%       Monia Ranalli
%       Department of Statistics
%       Sapienza University, Rome, Italy
%       Email : monia.ranalli@uniroma1.it
%
% OUTPUT__________________________________
% maximized complete data composite log-likelihood for models 1-8
% INPUT___________________________________
% param: a vector containing the parameters to estimate
% th_idx: the vector keeping track to which variable the thresholds belongs to
% X: sample data
% T: classification matrix
% P: the number of variables
% G: the number of groups
% Q: the number of factors
% j: indexes for varying thresholds (each variables has the first two thresholds set to 0 and 1, respectively)
% type: model type (1-8), that is CCC, CCU, CUC, CUU, UCC, UCU, UUC, UUU
% idxVar: 1 if ordinal variable, 0 if continuous
% pairsindex: indexes for the pairs of variables
% idLlog: if 0 estimate the parameters, if 1 parameter is 0
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
if (type==1),
    L0=param((P*G+length(th_idx)-2*O+1):(P*G+length(th_idx)-2*O+P*Q-Q*(Q-1)/2));
    L00=zeros(1,P*Q);
    L00(idLlog==0)=L0;
    L=reshape(L00,P,Q);
    psi=param((P*G+length(th_idx)-2*O+P*Q-Q*(Q-1)/2+1):(P*G+length(th_idx)-2*O+P*Q-Q*(Q-1)/2+1));
    J=L*L'+diag(repmat(psi,[P,1]));
    [ei, eig1]=eig(J);
    eig1(eig1<0)=0.001;
    Sig=ei*eig1*ei';
    Sig=repmat(Sig,[1,1,G]);
elseif (type==2),
    L0=param((P*G+length(th_idx)-2*O+1):(P*G+length(th_idx)-2*O+P*Q-Q*(Q-1)/2));
    L00=zeros(1,P*Q);
    L00(idLlog==0)=L0;
    L=reshape(L00,P,Q);
    psi=param((P*G+length(th_idx)-2*O+P*Q-Q*(Q-1)/2+1):(P*G+length(th_idx)-2*O+P*Q-Q*(Q-1)/2+P));
    J=L*L'+diag(psi);
    [ei, eig1]=eig(J);
    eig1(eig1<0)=0.001;
    Sig=ei*eig1*ei';
    Sig=repmat(Sig,[1,1,G]);
elseif (type==3),
    L0=param((P*G+length(th_idx)-2*O+1):(P*G+length(th_idx)-2*O+P*Q-Q*(Q-1)/2));
    L00=zeros(1,P*Q);
    L00(idLlog==0)=L0;
    L=reshape(L00,P,Q);
    psi=param((P*G+length(th_idx)-2*O+P*Q-Q*(Q-1)/2+1):(P*G+length(th_idx)-2*O+P*Q-Q*(Q-1)/2+G));
    for g=1:G,
        J=L*L'+diag(repmat(psi(g),[P,1]));
        [ei, eig1]=eig(J);
        eig1(eig1<0)=0.001;
        Sig(:,:,g)= ei*eig1*ei';
    end
elseif (type==4),
    L0=param((P*G+length(th_idx)-2*O+1):(P*G+length(th_idx)-2*O+P*Q-Q*(Q-1)/2));
    L00=zeros(1,P*Q);
    L00(idLlog==0)=L0;
    L=reshape(L00,P,Q);
    psi=reshape(param((P*G+length(th_idx)-2*O+P*Q-Q*(Q-1)/2+1):(P*G+length(th_idx)-2*O+P*Q-Q*(Q-1)/2+P*G)),P,G);
    for g=1:G,
        J=L*L'+diag(psi(:,g));
        [ei, eig1]=eig(J);
        eig1(eig1<0)=0.001;
        Sig(:,:,g)= ei*eig1*ei';
    end
elseif (type==5),
    L0=param((P*G+length(th_idx)-2*O+1):(P*G+length(th_idx)-2*O+P*Q*G-G*Q*(Q-1)/2));
    L00=zeros(1,P*Q*G);
    L00(idLlog==0)=L0;
    L=reshape(L00,P,Q,G);
    psi=param((P*G+length(th_idx)-2*O+P*Q*G-G*Q*(Q-1)/2+1):(P*G+length(th_idx)-2*O+P*Q*G-G*Q*(Q-1)/2+1));
    for g=1:G,
        J=L(:,:,g)*L(:,:,g)'+diag(repmat(psi,[P,1]));
        [ei, eig1]=eig(J);
        eig1(eig1<0)=0.001;
        Sig(:,:,g)= ei*eig1*ei';
    end
elseif (type==6),
    L0=param((P*G+length(th_idx)-2*O+1):(P*G+length(th_idx)-2*O+P*Q*G-G*Q*(Q-1)/2));
    L00=zeros(1,P*Q*G);
    L00(idLlog==0)=L0;
    L=reshape(L00,P,Q,G);
    psi=param((P*G+length(th_idx)-2*O+P*Q*G-G*Q*(Q-1)/2+1):(P*G+length(th_idx)-2*O+P*Q*G-G*Q*(Q-1)/2+P));
    for g=1:G,
        J=L(:,:,g)*L(:,:,g)'+diag(psi);
        [ei, eig1]=eig(J);
        eig1(eig1<0)=0.001;
        Sig(:,:,g)= ei*eig1*ei';
    end
elseif (type==7),
    L0=param((P*G+length(th_idx)-2*O+1):(P*G+length(th_idx)-2*O+P*Q*G-G*Q*(Q-1)/2));
    L00=zeros(1,P*Q*G);
    L00(idLlog==0)=L0;
    L=reshape(L00,P,Q,G);
    psi=param((P*G+length(th_idx)-2*O+P*Q*G-G*Q*(Q-1)/2+1):(P*G+length(th_idx)-2*O+P*Q*G-G*Q*(Q-1)/2+G));
    for g=1:G,
        J=L(:,:,g)*L(:,:,g)'+diag(repmat(psi(g),[P,1]));
        [ei, eig1]=eig(J);
        eig1(eig1<0)=0.001;
        Sig(:,:,g)= ei*eig1*ei';
    end
elseif (type==8),
    L0=param((P*G+length(th_idx)-2*O+1):(P*G+length(th_idx)-2*O+P*Q*G-G*Q*(Q-1)/2));
    L00=zeros(1,P*Q*G);
    L00(idLlog==0)=L0;
    L=reshape(L00,P,Q,G);
    psi=reshape(param((P*G+length(th_idx)-2*O+P*Q*G-G*Q*(Q-1)/2+1):(P*G+length(th_idx)-2*O+P*Q*G-G*Q*(Q-1)/2+P*G)),P,G);
    for g=1:G,
        J=L(:,:,g)*L(:,:,g)'+diag(psi(:,g));
        [ei, eig1]=eig(J);
        eig1(eig1<0)=0.001;
        Sig(:,:,g)=ei*eig1*ei';
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
