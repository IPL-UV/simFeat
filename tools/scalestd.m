
function  x = scalestd(x)

[NumSamples,NumBands] = size(x);

for j=1:NumBands
   mu=mean(x(:,j));
   sig=std(x(:,j));
   x(:,j)=(x(:,j)-mu)/sig;
end
