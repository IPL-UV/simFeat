% Binarize of a vector
%
% Inputs:
%       -Y : Vector, M(samples)x1.
%
% Outputs:
%       -Yb: Matrix, M(samples)xC(Number of values as in Y without no repetitions). 
%

function YY = binarize(Y)

unics = unique(Y);
YY = zeros(length(Y),length(unics));
for i=1:length(unics)
	YY(find(Y==unics(i)),i) = 1;
end;


