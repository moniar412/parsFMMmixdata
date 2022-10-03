function p = bvnP( xl, xu, yl, yu, r )
%BVN
%  A function for computing bivariate normal probabilities.
%  bvn calculates the probability that 
%    xl < x < xu and yl < y < yu, 
%  with correlation coefficient r.
%   p = bvn( xl, xu, yl, yu, r )
%

%   Author
%       Alan Genz, Department of Mathematics
%       Washington State University, Pullman, Wa 99164-3113
%       Email : alangenz@wsu.edu
%
p=ones(length(xl),1);
for i=1:length(xl)
  p(i) = bvnlr(xl(i),yl(i),r) - bvnlr(xu(i),yl(i),r) - bvnlr(xl(i),yu(i),r) + bvnlr(xu(i),yu(i),r); 
  p(i) = max( eps, min( p(i), 1 ) );
end
  %
%   end bvn
%