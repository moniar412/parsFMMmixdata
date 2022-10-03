function [U]=randU(n,G)
%
%   Author
%       Roberto Rocci
%       Department of Statistics
%       Sapienza University, Rome, Italy
%       Email : roberto.rocci@uniroma1.it
%
% generates a random fuzzy partition of n subjects in g groups
%
U=rand(n,G);
U=repmat(1./sum(U,2),1,G).*U;
