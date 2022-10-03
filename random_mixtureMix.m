function [X,d,npg]= random_mixtureMix (N, mu, sigma, pg,br,th_idx,G,P,idxO,st)
%
%   Author
%       Monia Ranalli
%       Department of Statistics
%       Sapienza University, Rome, Italy
%       Email : monia.ranalli@uniroma1.it
% OUTPUT__________________________________
% X: random generation of a multivariate gaussian mixture
% INPUT___________________________________
% N: the sample size
% mu: a matrix with G mean vectors
% Sigma: an array of dimension PxPXG of variance/covariance matrices for
% each group G
% pg: the vector containing the probabilities associated with each group
% br: the vector of thresholds 
% th_idx: a vector keeping track to which variable the thresholds belongs to
% G: the number of groups
% P: the number of variables
% idxO: indexes of ordinal variables
% st: seed
%
rng(st,'twister');
pg=pg/sum(pg);
n.samp = randsample(1:G,N,true,pg);
v=unique(n.samp); n.pg=hist(n.samp,v);
npg=n.pg;
c=cell(G,1);
for g=1:G,
     c{g,1}=mvnrnd(mu(g,:),sigma(:,:,g),n.pg(g));
end
d=cell2mat(c);
mu1 = reshape(mu',P,G,1);
mug=(mu1*pg')';
sigma1=reshape((reshape(sigma,P*P,G).*repmat(pg,P*P,1)),P,P,G);
var.W=sum(sigma1,3);
m=reshape((mu1'-repmat(mug,G,1))',P,G);
var.B=sum((m.^2).*repmat(pg,P,1),2);
var.G=(diag(var.W)+var.B)';
O=length(idxO);
b=cell(O,1);
for i=1:O,
    b{i,1}=[-Inf br(th_idx==i) Inf];
end
X=d;
for i=1:O,
    X(:,idxO(i))=ordinal(d(:,idxO(i)),[],[],b{i,1});
end
end








