function  x = scalestd(x)

for j = 1:size(x,2)
    mu = mean(x(:,j));
    sig = std(x(:,j));
    x(:,j) = (x(:,j) - mu) / sig;
end
