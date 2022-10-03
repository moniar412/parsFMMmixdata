function [U,Xmean,res]=km(X,U0)
%
%   Author
%       Roberto Rocci
%       Department of Statistics
%       Sapienza University, Rome, Italy
%       Email : roberto.rocci@uniroma1.it
%
%--------------------
% k-means algorithm |
%--------------------
%
% n = number of objects
% k = number of clusters of the partition
% maxiter = max number of iterations
% start=1 means start fuzzy
%
maxiter=99;
[n,k]=size(U0);
un=ones(n,1);
uk=ones(k,1);
dist=zeros(k,n);
eps=0.0000000001;
st=sum(sum(X.^2));
so=0;
Xuk=kron(X,uk);
%
% start fuzzy
% modificato da 24 a 27
% if start==1,
%    [U]=kmeansf(X,U0,1.1);
%    U0=(U==(uk*max(U'))');
% end
%
Xmean0 = pinv(U0)*X;
% 
for iter=1:maxiter
   %
   % given Xmean0 assign each units to the closest cluster
   dist(:)=sum( ( ( Xuk-kron(un,Xmean0) ).^2 )' );
   U=(dist==uk*min(dist))';
   if (sum(sum(U))~=n),
       U=zeros(n,k);
       for i=1:n,
           [m,p]=min(dist(:,i));
           U(i,p)=1;
       end
   end
   su=sum(U);
   while sum(su==0)>0,
       [m,p1]=min(su);
       [m,p2]=max(su);
       ind=find(U(:,p2));
       ind=ind(1:floor(su(p2)/2));
       U(ind,p1)=1;
       U(ind,p2)=0;
       su=sum(U);
   end 
   %
   % given U compute Xmean (compute centroids)
   Xmean = diag(1./su)*U'*X;
   %   
   % stopping rule
   sa=100*sum(sum((U*Xmean).^2))/st;
   dif=sa-so;
   if (dif > eps)&(sum(sum(abs(U-U0)))>1)  
      Xmean0=Xmean;
      U0=U;
      so=sa;
   else
      pF=(sa/(k-1))/((100-sa)/(n-k));
%       disp(sprintf('k-means: sse=%g, pF=%g, iter=%g, dif=%g',sa,pF,iter,dif))
      break
   end   
end
res=100-sa;