function [X,Y] = generate_toydata(n,method)

% [X,Y] = generate_toydata(n,method)
%
% Generates toy classification problems.
%
% INPUTS:
%    n     : points in class +1 and n points in class 2
%    method:
%       'lines'          : Parallel lines
%       'ellipsoids'     : Parallel ellipsoids
%       'moons'          : Two moons
%       'swiss'          : 2d swiss roll 
%
% Gustavo Camps-Valls, 2007(c)
% gcamps@uv.es

[X,Y] = feval(method,n);

function [X,Y] = wheel(n)

        [X,Y] = pinwheel(0.5, 0.3, 2, n, 1);

function [X,Y] = linearmixing(n)

        s = sin(0:(50/n):50);
        N = length(s);
        n = [zeros(1,round(N/2)) rand(1,round(N/2)-1)];
        M = [1.0 0.5; 
             0.2 0.8];
        X = M * [s;
                 n];
        X = X';
        Y = 0.5*sign(X(:,1))+1.5;

function [X,Y] = noisysinus(n)

        t = 0:(2*pi/n):2*pi;
        s  = sin(t).*t + 0.3*randn(1,n+1);
        s2 = sin(t).*t + 0.3*randn(1,n+1) -1;
        X = [t', s';
             t', s2'];
        Y = [ones(n+1,1); 2*ones(n+1,1)];

function [X,Y] = lines(n)

        var = 1;
        x = 0.015;
        y = 0.015;
        X1=[randn(n,1)*0.01+x       randn(n,1)*0.01+y];
        Y1=ones(n,1);
        X2=[randn(n,1)*0.01-x       randn(n,1)*0.01-y];
        Y2=2*ones(n,1);
        X=[X1;X2];
        Y=[Y1;Y2];

function [X,Y] = xor(n);

        n = n/2;
        X1 = mvnrnd([1;20],[1 0;0 1],n);
        Y1 = ones(n,1);
        X2 = mvnrnd([1;1],[1 0;0 1],n);
        Y2 = 2*ones(n,1);
        X3 = mvnrnd([20;1],[1 0;0 1],n);
        Y3 = ones(n,1);
        X4 = mvnrnd([20;20],[1 0;0 1],n);
        Y4 = 2*ones(n,1);
        X  = [X1;X2;X3;X4];
        Y  = [Y1;Y2;Y3;Y4];

function [X,Y] = noisyxor(n);

        n = n/2;
        X1 = mvnrnd([1;10],[.01 0;0 2],n);
        Y1 = ones(n,1);
        X2 = mvnrnd([1;1],[.01 0;0 2],n);
        Y2 = 2*ones(n,1);
        X3 = mvnrnd([10;1],[.01 0;0 2],n);
        Y3 = ones(n,1);
        X4 = mvnrnd([10;10],[.01 0;0 2],n);
        Y4 = 2*ones(n,1);
        X  = [X1;X2;X3;X4];
        Y  = [Y1;Y2;Y3;Y4];

function [X,Y] = balls3(n);

        n = n/2;
        X1 = mvnrnd([15 15],[1 0;0 1],n);
        Y1 = ones(n,1);
        X2 = mvnrnd([10;10],[1 0;0 1],n);
        Y2 = 2*ones(n,1);
        X3 = mvnrnd([15;1],[1 0;0 1],n);
        Y3 = 3*ones(n,1);
        X  = [X1;X2;X3];
        Y  = [Y1;Y2;Y3];

function [X,Y] = parabolloid(n);

        space=1.5;
        noise = 0.05;
        r=randn(n,1)*noise+1;
        theta=randn(n,1)*pi;
        r1=1.1*r; r2=r;
        X=([r1.*cos(theta) abs(r2.*sin(theta))]);
        Y=ones(n,1);

function [X,Y] = ellipsoids(n_points)
		
        mean1 = [0;1];
        mean2 = [0;3];
        cov = [1 .01; .01 1];
        
		X1 = mvnrnd(mean1,cov,n_points);
		X2 = mvnrnd(mean2,cov,n_points);
		
		X = [X1;X2];
		media = mean(X);
		X = X - ones(2*n_points,1)*mean(X);
		Y = zeros(2*n_points,1); 
        Y(1:n_points,1) = 1; 
        Y(n_points+1:end,1) = 2;
		

function [X,Y] = ellipsoids3(n_points)
		
        mean1 = [-2;1];
        mean2 = [2;-2];
        mean3 = [4;2];
        cov = [1 .9; .9 1];
        
		X1 = mvnrnd(mean1,cov,n_points);
		X2 = mvnrnd(mean2,cov,n_points);
		X3 = mvnrnd(mean3,cov,n_points);

        Y1 = ones(n_points,1);
        Y2 = 2*ones(n_points,1);
        Y3 = 3*ones(n_points,1);

		X = [X1;X2;X3];
		Y = [Y1;Y2;Y3];

function [X,Y] = moons(n)

        space = 2;
        noise = 0.05;
        r=randn(n,1)*noise+1;
        theta=randn(n,1)*pi;
        r1=1.1*r; r2=r;
        X1=([r1.*cos(theta) abs(r2.*sin(theta))]);
        Y1=ones(n,1);

        r=randn(n,1)*noise+1;
        theta=randn(n,1)*pi+2*pi;
        r1=1.1*r; r2=r;
        X2=([r1.*cos(theta)+space*rand -abs(r2.*sin(theta)) + 0.6 ]);
        Y2=2*ones(n,1);

        X=[X1;X2];
        Y=[Y1;Y2];

function [X,Y] = swiss(n)

        tt = (3*pi/2)*(1+2*rand(1,n));
        X1 = [tt.*cos(tt); tt.*sin(tt)]';

        tt = (3*pi/2)*(1+2*rand(1,n));
        X2 = [tt.*cos(tt);  tt.*sin(tt)]'; X2=1.6*X2;
        X=[X1;X2]/25+randn(2*n,2)*0.03;
        Y=[ones(n,1); 2*ones(n,1)];
