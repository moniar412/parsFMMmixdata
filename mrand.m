function mri=mrand(N)
%
%   Author
%       Roberto Rocci
%       Department of Statistics
%       Sapienza University, Rome, Italy
%       Email : roberto.rocci@uniroma1.it
% OUTPUT__________________________________
% modified rand index (Hubert & Arabie 1985, JCGS p.198)
% INPUT___________________________________
% confusion matrix between two partitions (example true and fitted
% partitions)
n=sum(sum(N));
sumi=.5*(sum(sum(N').^2)-n);
sumj=.5*(sum(sum(N).^2)-n);
pb=2*sumi*sumj/(n*(n-1));
mri=(.5*(sum(sum(N.^2))-n)-pb)/((sumi+sumj)/2-pb);