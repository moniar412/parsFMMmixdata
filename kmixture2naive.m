function [theta_init,T0,fval]=kmixture2naive(X,th_idx,P,G,O,idxO)
%
%   Author
%       Monia Ranalli
%       Department of Statistics
%       Sapienza University, Rome, Italy
%       Email : monia.ranalli@uniroma1.it
%
% OUTPUT__________________________________
% parameter initialization and initial classification matrix
% INPUT___________________________________
% X: sample data
% th_idx: a vector keeping track to which variable the thresholds belongs to
% P: the number of variables
% G: the number of groups
% O: the number of ordinal variables
% idxO: index of ordinal variables
no.pairs=O*(O-1)/2;
N=size(X,1);
U=randU(size(X,1),G);
[U,~,~]=km(X,U);
[out,Mmu,MOmeQ,MOmel,~,~,~,~,~,~]=mixhet(X,U,10^-6,1);
p0=sum(out)/size(X,1);
mu=Mmu;
variance=zeros(G,P);
R1=cell(G,1);
cors=zeros(G,P*(P-1)/2);
sigma=cell(G,1);
ch=zeros(G,P*(P+1)/2);
r=find(tril(ones(P),-1));
for g=1:G,
    variance(g,:)=diag(MOmeQ(:,(g-1)*P+(1:P))*diag(MOmel(:,g))* MOmeQ(:,(g-1)*P+(1:P))');
    sigma{g}=MOmeQ(:,(g-1)*P+(1:P))*diag(MOmel(:,g))* MOmeQ(:,(g-1)*P+(1:P))';
    v=chol(sigma{g});v=v';
    ch(g,:)=v(find(tril(ones(size(v)),0)))';
    [~, R1{g}]=cov2corr(sigma{g});
    cors(g,:)=R1{g}(r');
end
var=variance;
% to compute the thresholds
th=cell(1,O);
fval=0;
for j=1:O,
    c=unique(X(:,idxO(j)));
    h=hist(X(:,idxO(j)),c)';
    cumh=cumsum(h);
    cumrel=bsxfun(@rdivide,cumh,cumh(size(cumh,1),:));
    if length(cumrel)==length(th_idx(th_idx==j)),
        cumrel(size(cumrel,1),:)=[];
        cumrel=[cumrel; cumrel(size(cumrel,1))+rand(1)];
        cumrel1=reshape(cumrel,1,[]);
    elseif length(cumrel)<length(th_idx(th_idx==j)),
        cumrel(size(cumrel,1),:)=[];
        cumrel=[cumrel' cumrel(size(cumrel,1))+rand(1,length(th_idx(th_idx==j))-size(cumrel,1))];
        cumrel=cumsum(cumrel/(sum(cumrel)));
        cumrel1=reshape(cumrel,1,[]);
    else
        cumrel(size(cumrel,1),:)=[];
        cumrel1=reshape(cumrel,1,[]);
    end
    th{j}=zeros(1,length(th_idx(th_idx==j)));
    x0=sort(normrnd(0,1,1,length(th_idx(th_idx==j))));
    for i=1:length(th_idx(th_idx==j)),
        [th{j}(i), val]=fminsearch(@(x)funct(x,p0,mu(:,j),var(:,j),cumrel1,G,i),x0(i));
        fval=fval+val;
    end
    th{j}=  sort(th{j});
    mu(:,idxO(j))=mu(:,idxO(j))-th{j}(1);
end
th=cell2mat(th);
gamma=th;
T0=out;
for i=1:O,
    ga=sort(gamma(th_idx==i));
    ga=(ga-ga(1))/(ga(2)-ga(1));
    gamma(th_idx==i)=ga;
end
theta_init=struct('pg',p0,'mu',mu,'sigma',var,'choleg',ch,'cors',cors,'gamma',gamma);
end



