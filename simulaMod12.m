% Data generation Model type 12 - (SC)UU
%   Author
%       Monia Ranalli
%       Department of Statistics
%       Sapienza University, Rome, Italy
%       Email : monia.ranalli@uniroma1.it
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% INPUTS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% number of groups G, number of variables P, number of factor Q, number of
% ordinal variables O, model type 12 - (SC)UU, sample size N, mixture
% weights pg
G=3; P=8; Q=4; O=4; type=12; N=500; pg=[.25 .35 .4];
% 1 if ordinal, 0 if continuous
idxVar=[1 1 1 1 0 0 0 0 ];
% index for ordinal variables
idxO=[1 2 3 4];
% pairs index for ordinal variables
pairs.index=combnk(1:O,2)';
if O>3 pairs.index=fliplr(pairs.index); end
% indexes for select parameter estimates of matrix L
V=reshape(1:(P*Q*G),P,Q,G);
idL0=[];
for g=1:G,
    Vu=(triu(V(1:Q,1:Q,g),1));
    idL0=[idL0,(Vu(Vu>0))'];
end
idLall=1:(P*Q*G);
%if 0 estimate the parameters, if 1 parameter is 0
idLlog=ismember(idLall,idL0);
%
% a possible simulation study for a number of samples nk
%
% creation of vectors for storing ari values and Euclidean distances
% between parameter estimates and true values
ari=ones(nk,1);
SobsC=ones(1,nk);
So=ones(1,nk);
Sg=ones(1,nk);
Spg=ones(1,nk);
SL=ones(1,nk);
Sl=ones(1,nk);
Spsi=ones(1,nk);
% set the seed
st0=1;
for rep=1:nk,
    st=st0*nk+1+rep;
    rng(st,'twister');
% data generation given L, Psi, mu and thresholds br
L=unifrnd(-1,1,P,Q);
L0=zeros(Q,Q,G);
L0(:,:,1)=eye(Q);
l0true=ones(G,Q);
for g=2:G,
    l0true(g,:)=unifrnd(0,2,1,Q);
    L0(:,:,g)=diag(l0true(g,:));
end
Psi=unifrnd(0,1,P,G);
PSI=ones(P,P,G);
for g=1:G,
    PSI(:,:,g)=diag(Psi(:,g));
end
mu=[0.5, 0.5,1,1,-1,2,-2,-1;1.5,1.5,0,0,1,0,2,0;
    0.5 -0.5 -1 -1 0 -2 0 -1];
br=repmat([0 1 1.5:1:2.5],1, O);
th_idx=repmat(1:O,4,1);
th_idx=th_idx(:)';
% number of categories for each ordinal variable
Sig=ones(P,P,G);
for g=1:G,
    Sig(:,:,g)=L*L0(:,:,g)*L'+PSI(:,:,g);
end
% data generation
    [X,d,npg]= random_mixtureMix (N, mu, Sig, pg,br,th_idx,G,P,idxO,st);
% parameter initialization
   theta_init=cell(1,30);
    T0=cell(1,30);
    fval=ones(1,30);
    for r=1:30,
        [theta_init{r},T0{r},fval(r)]=kmixture2naive(X,th_idx,P,G,O,idxO);
    end
    idxg=find((fval)==min(fval));
    theta=theta_init{idxg};
    T0=T0{idxg};
    S=ones(P,P,G);
    for g=1:G,
        L1=ones(P,P);
        L1(tril(L1,0)~=0)=theta.choleg(g,:)';
        L1=tril(L1,0);
        S(:,:,g)=L1*L1';
    end
     Lh=ones(P,Q,G);
    PsiH=ones(P,P,G);
    Psih=unifrnd(0,1,P,G);
    ps=ones(P,G);
     [nt ntc]=find(isnan(T0));
    T0(nt,:)=repmat(1/G,size(nt,1),G);
    theta.psi=ps;
    theta.L=Lh;
    for g=1:G,
        [e ee]=eig(S(:,:,g));
        ee(ee<0)=0.001;
        S(:,:,g)=e*ee*e';
        [Lh(:,:,g),ps(:,g)] = factoran(S(:,:,g),Q,'Xtype','covariance');
    end
    L0hat=ones(G,Q);
    Lhh=mean(Lh,3);
    for g=1:G,    
        psih=diag(diag(S(:,:,g)-Lhh*diag(L0hat(g,:))*Lhh'));
        psih(psih<0)=unifrnd(0,1);
        PsiH(:,:,g)=psih;
        ps(:,g)=diag(psih);
    end
    theta.L=Lhh;
    theta.psi=ps;
    theta.l0=L0hat;
    theta_init=theta;
    % EM like algorithm for fitting (SC)UU model
    [T, theta,plimix,lik,er0]=EMparsMixs(theta_init,T0,10^-5,X,th_idx,G,P,Q,type,idxVar);
    % classification
    post=sum(plimix,3);
    post=repmat(1./sum(post,2),1,G).*post;
    % hard classification
    posth=ftoh(post);
    % true classification
    out_true=[repmat([1 0 0],npg(1),1);repmat([0  1 0],npg(2),1);
        repmat([0  0 1],npg(3),1)];
    % ari estimation
    ari(rep)=mrand(posth'*out_true);
    % sort parameter estimations to avoid label switching
    [spg,idxs]=sort(theta.pg);
    muh=theta.mu(idxs,:);
    gammah=theta.gamma;
    psih=theta.psi(:,idxs);
    Lh=theta.L;
    l0=theta.l0(idxs,:);
    muhc=muh(:,idxVar==0);
    muc=mu(:,idxVar==0);
    muho=muh(:,idxVar==1);
    muo=mu(:,idxVar==1);
    % computation of Euclidean squared distance between the estimates 
    % and the true parameter values
    SobsC(rep)=mean((muhc(:)-muc(:)).^2);
    So(rep)=mean((muho(:)-muo(:)).^2);
    Sg(rep)=mean((gammah(gammah>1)-br(br>1)).^2);
    Spg(rep)=mean((spg(1:G-1)-pg(1:G-1)).^2);
    Lhvec=(Lh(:));
    Lvec=(L(:));
    l0vec=l0(2:end,:);
    l0vec=l0vec(:);
    l0truevec=l0true(2:end,:);
    l0truevec=l0truevec(:);
    SL(rep)=mean((Lhvec-Lvec).^2);
    Sl(rep)=mean((l0vec-l0truevec).^2);
    Spsi(rep)=mean((psih(:)-Psi(:)).^2);
    acc=[SobsC(rep) So(rep) Sg(rep) Spg(rep) SL(rep) Sl(rep) Spsi(rep)];
    save('ARI500type12.txt','ari','-ascii','-append')
    save('accuracyN500type12.txt','acc','-ascii','-append')
end








