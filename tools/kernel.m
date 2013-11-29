%
% Compute a kernel given input data
%
% Inputs:
%         X:  training data (samples in rows, features in columns)
%         X2: test data (samples in rows, features in columns)
%         param: kernel free parameter
%         type: type of kernel function, i.e. linear, polynomial, rbf
% Output:
%         K:  kernel matrix
%

function K = kernel(type,X,X2,param)

if strcmp(type,'linear')
    
    K = X*X2';
    
elseif strcmp(type,'poly')
    
    K = (1+X*X2').^param;
    
elseif strcmp(type,'rbf')
    
    [n d] = size(X);
    [n2 d] = size(X2);
    for i=1:n
        for j =1:n2
            D(i,j) = norm(X(i,:) - X2(j,:));
        end
    end
    K = exp(-D.^2/(2*param^2));
   
    
else
    disp('unspecified kernel function.')
end
    
    