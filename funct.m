function f=funct(x,p0,mu,var,cumrel1,G,i)
%
%   Author
%       Monia Ranalli
%       Department of Statistics
%       Sapienza University, Rome, Italy
%       Email : monia.ranalli@uniroma1.it
%
% the function minimizes the quadratic difference between theoretical and
% empirical probabilities
%
f=0;
for g=1:G,
     f=f+p0(g)*normcdf(x,mu(g),sqrt(var(g)));
end
f=(f-cumrel1(i))^2;
end


