% Unbinarize of a vector
%
% Inputs:
%       - Y  : Vector, M(samples) x 1
%
% Outputs:
%       - Yb : Matrix, M(samples) x C(Number of values as in Y without repetitions)
%
%
% Warning: whis function assumess that classes are numbered from 1 to C,
%          and that all classes exist

function Y = unbinarize(Yb)

[~, Y] = max(Yb,[],2);
