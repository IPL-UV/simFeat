function array = disp_patches(A,m)

% DISP_PATCHES displays the image basis functions at the columns of matrix A
%
% array = disp_patches(A,m)
%
%  Usage:
%   A = basis function matrix
%   m = number of rows (of the image blocks)

[L M] = size(A);

sz = sqrt(L);

buf = 1;

if ~exist('m','var'),
    if floor(sqrt(M))^2 ~= M
        m = floor(sqrt(M / 2));
        n = M / m;
    else
        m = sqrt(M);
        n = m;
    end
else
    n = M / m;
end

array = -ones(buf + m * (sz + buf), buf + n * (sz + buf));

k = 1;

for i = 1:m
    for j = 1:n
        clim = max(abs(A(:,k)));
        array(buf + (i-1) * (sz+buf) + (1:sz), buf + (j-1) * (sz+buf) + (1:sz)) = ...
            reshape(A(:,k),sz,sz) / clim;
        k = k + 1;
    end
end

imagesc(array, 'EraseMode', 'none', [-1 1]);
axis image off

drawnow
