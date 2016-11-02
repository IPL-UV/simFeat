% Compute a kernel given input data
%
% Inputs:
%         X:  training data (samples in rows, features in columns)
%         X2: test data (samples in rows, features in columns)
%         param: kernel free parameter
%         type: type of kernel function, i.e. linear, polynomial, rbf
% Output:
%         K:  kernel matrix

function K = kernel(type,X1,X2,param)

if strcmp(type,'linear')
    
    K = X1 * X2';
    
elseif strcmp(type,'poly')
    
    K = (1 + X1 * X2').^param;
    
elseif strcmp(type,'rbf')
    
    n1 = size(X1,1);
    n2 = size(X2,1);
    D = zeros(n1,n2);
    for i = 1:n1
        for j = 1:n2
            D(i,j) = norm(X1(i,:) - X2(j,:));
        end
    end
    % Ugly fix: param (sigma) could be a vector when using estimateSigma with a method
    % like 'quantiles', and this will fail:
    %K = exp(-D.^2 / (2*param^2));
    % For now we just workaround the problem:
    K = exp(-D.^2 / (2*mean(param)^2));
   
else
    disp('kernel function not implemented')
end
