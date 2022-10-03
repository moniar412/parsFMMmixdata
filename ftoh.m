function Uh=ftoh(Uf)
%
%   Author
%       Roberto Rocci
%       Department of Statistics
%       Sapienza University, Rome, Italy
%       Email : roberto.rocci@uniroma1.it
%
% converts a partition from fuzzy to hard
%
[n,nk]=size(Uf);
Uh=zeros(n,nk);
[a,ind]=max(Uf');
for i=1:n
    Uh(i,ind(i))=1;
end