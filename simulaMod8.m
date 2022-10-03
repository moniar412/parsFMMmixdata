% Data generation Model type 8 - UUU
%   Author
%       Monia Ranalli
%       Department of Statistics
%       Sapienza University, Rome, Italy
%       Email : monia.ranalli@uniroma1.it
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% INPUTS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% number of groups G, number of variables P, number of factor Q, number of
% ordinal variables O, model type 8 - UUU, sample size N, mixture
% weights pg
G=2; P=8; Q=4; O=4; type=8; N=500; pg=[.3 .7];
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
ari=ones(1,nk);
SobsC=ones(1,nk);
So=ones(1,nk);
Sg=ones(1,nk);
Spg=ones(1,nk);
SL=ones(1,nk);
Spsi=ones(1,nk);
% set the seed
st0=1;
for rep=1:nk,
    st=st0*nk+1+rep;
    rng(st,'twister');
    % data generation given L, Psi, mu and thresholds br
    L=unifrnd(-1,1,P,Q,G);
    for g=1:G,
        dl=diag(L(1:Q,1:Q,g));
        dl=abs(dl);
        dL=L(1:Q,1:Q,g);
        dLvec=dL(:);
        idx=find(dL==diag(dL));
        dLvec(idx)=dl;
        L(1:Q,1:Q,g)=reshape(dLvec,Q,Q);
        L(1:Q,1:Q,g)= tril(L(1:Q,1:Q,g));
    end
    Psi=unifrnd(0,1,P,G);
    PSI=ones(P,P,G);
    for g=1:G,
        PSI(:,:,g)=diag(Psi(:,g));
    end
    br=repmat([0 1  2],1, O);
    th_idx=repmat(1:O,3,1);
    th_idx=th_idx(:)';
    Sig=ones(P,P,G);
    for g=1:G,
        Sig(:,:,g)=L(:,:,g)*L(:,:,g)'+PSI(:,:,g);
    end
    mu=[0.5, 0.5,1,1,-1,2,-2,-1;1.5,1.5,0,0,1,0,2,0];
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
        [q ~]=qr(Lh(1:Q,1:Q,g)');
        Lh(:,:,g)= Lh(:,:,g)*q;
        psih=diag(diag(S(:,:,g)-Lh(:,:,g)*Lh(:,:,g)'));
        psih(psih<0)=unifrnd(0,1);
        PsiH(:,:,g)=psih;
        ps(:,g)=diag(psih);
    end
    theta.L=Lh;
    theta.psi=ps;
    theta_init=theta;
    % EM like algorithm for fitting UUU model
    [T, theta,plimix,lik,er0]=EMparsMix(theta_init,T0,10^-5,X,th_idx,G,P,Q,type,idxVar);
    % classification
    post=sum(plimix,3);
    post=repmat(1./sum(post,2),1,G).*post;
    % hard classification
    posth=ftoh(post);
    % true classification
    out_true=[repmat([1 0],npg(1),1);repmat([0  1],npg(2),1)];
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
    SL(rep)=mean((Lhvec(idLlog==0)-Lvec(idLlog==0)).^2);
    Spsi(rep)=mean((psih(:)-Psi(:)).^2);
    acc=[SobsC(rep) So(rep) Sg(rep) Spg(rep) SL(rep) Spsi(rep)];
    save('ARI500type8.txt','ari','-ascii','-append')
    save('accuracyN500type8.txt','acc','-ascii','-append')
end




