% Binarize of a vector
%
% Inputs:
%       - Y  : Vector, M(samples) x 1
%
% Outputs:
%       - Yb : Matrix, M(samples) x C(Number of values as in Y without repetitions)
%

function YY = binarize(Y)

unics = unique(Y);
YY = zeros(length(Y),length(unics));
for i = 1:length(unics)
    YY(Y == unics(i), i) = 1;
end
