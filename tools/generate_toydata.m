function [X,Y] = generate_toydata(n,method)

% [X,Y] = generate_toydata(n,method)
%
% Generates toy classification problems.
%
% INPUTS:
%    n     : points in class 1 and n points in class 2
%    methods:
%       'wheel'          : "pinewheel" dataset
%       'linearmixing'   : 
%       'noisysinus'     : Noisy sinousiods
%       'xor'            : XOR function
%       'noisyxor'       : Noisy XOR
%       'balls3'
%       'ellipsoids'     : Ellipsoids
%       'ellipsoids3'    : Ellipsoids 3-D
%       'lines'          : Parallel lines
%       'ellipsoids'     : Parallel ellipsoids
%       'moons'          : Two moons
%       'swiss'          : 2d swiss roll 
%
% Gustavo Camps-Valls, 2007(c)
% gcamps@uv.es
%
% JoRdI, 2016 (minor changes)

switch method
    case 'wheel'
        
        [X,Y] = pinwheel(0.5, 0.3, 2, n, 1);
        
    case 'linearmixing'
        
        s = sin(0:(50/n):50);
        N = length(s);
        n = [zeros(1,round(N/2)) rand(1,round(N/2)-1)];
        M = [1.0 0.5 ; 0.2 0.8];
        X = M * [s ; n];
        X = X';
        Y = 0.5 * sign(X(:,1)) + 1.5;

    case 'noisysinus'

        t = 0:(2*pi/n):2*pi;
        s  = sin(t) .* t + 0.3 * randn(1,n+1);
        s2 = sin(t) .* t + 0.3 * randn(1,n+1) - 1;
        X = [t', s' ; t', s2'];
        Y = [ones(n+1,1) ; 2*ones(n+1,1)];

    case 'lines'

        % var = 1;
        x = 0.015;
        y = 0.015;
        X1 = [randn(n,1) * 0.01 + x , randn(n,1) * 0.01 + y];
        Y1 = ones(n,1);
        X2 = [randn(n,1) * 0.01 - x , randn(n,1) * 0.01 - y];
        Y2 = 2 * ones(n,1);
        X = [X1 ; X2];
        Y = [Y1 ; Y2];

    case 'xor'

        n = n / 2;
        X1 = mvnrnd([1;20],[1 0;0 1],n);
        Y1 = ones(n,1);
        X2 = mvnrnd([1;1],[1 0;0 1],n);
        Y2 = 2 * ones(n,1);
        X3 = mvnrnd([20;1],[1 0;0 1],n);
        Y3 = ones(n,1);
        X4 = mvnrnd([20;20],[1 0;0 1],n);
        Y4 = 2 * ones(n,1);
        X  = [X1 ; X2 ; X3 ; X4];
        Y  = [Y1 ; Y2 ; Y3 ; Y4];

    case 'noisyxor'

        n = n / 2;
        X1 = mvnrnd([1;10],[.01 0;0 2],n);
        Y1 = ones(n,1);
        X2 = mvnrnd([1;1],[.01 0;0 2],n);
        Y2 = 2 * ones(n,1);
        X3 = mvnrnd([10;1],[.01 0;0 2],n);
        Y3 = ones(n,1);
        X4 = mvnrnd([10;10],[.01 0;0 2],n);
        Y4 = 2 * ones(n,1);
        X  = [X1 ; X2 ; X3 ; X4];
        Y  = [Y1 ; Y2 ; Y3 ; Y4];

    case 'balls3'

        n = n / 2;
        X1 = mvnrnd([15 15],[1 0;0 1],n);
        Y1 = ones(n,1);
        X2 = mvnrnd([10;10],[1 0;0 1],n);
        Y2 = 2 * ones(n,1);
        X3 = mvnrnd([15;1],[1 0;0 1],n);
        Y3 = 3 * ones(n,1);
        X  = [X1 ; X2 ; X3];
        Y  = [Y1 ; Y2 ; Y3];

    case 'parabolloid'

        % space = 1.5;
        noise = 0.05;
        r = randn(n,1) * noise + 1;
        theta = randn(n,1) * pi;
        r1 = 1.1 * r;
        r2 = r;
        X = ([r1 .* cos(theta) , abs(r2 .* sin(theta))]);
        Y = ones(n,1);

    case 'ellipsoids'
		
        mean1 = [0 ; 1];
        mean2 = [0 ; 3];
        cov = [1 .01 ; .01 1];
        
		X1 = mvnrnd(mean1, cov, n);
		X2 = mvnrnd(mean2, cov, n);
		
		X = [X1 ; X2];
		X = X - ones(2 * n, 1) * mean(X);
		Y = zeros(2 * n, 1); 
        Y(1:n, 1) = 1; 
        Y(n+1:end, 1) = 2;

    case 'ellipsoids3'
		
        mean1 = [-2 ; 1];
        mean2 = [2 ; -2];
        mean3 = [4 ; 2];
        cov = [1 .9 ; .9 1];
        
		X1 = mvnrnd(mean1, cov, n);
		X2 = mvnrnd(mean2, cov, n);
		X3 = mvnrnd(mean3, cov, n);

        Y1 = ones(n,1);
        Y2 = 2 * ones(n,1);
        Y3 = 3 * ones(n,1);

		X = [X1 ; X2 ; X3];
		Y = [Y1 ; Y2 ; Y3];

    case 'moons'

        space = 1;
        noise = 0.05;
        
        r = randn(n,1) * noise + 1;
        theta = randn(n,1) * pi;
        r1 = 1.1 * r;
        r2 = r;
        X1 = ([r1 .* cos(theta) , abs(r2 .* sin(theta))]);
        Y1 = ones(n,1);

        r = randn(n,1) * noise + 1;
        theta = randn(n,1) * pi + 2 * pi;
        r1 = 1.1 * r;
        r2 = r;
        X2 = ([r1 .* cos(theta) + space * rand , -abs(r2 .* sin(theta)) + 0.6]);
        Y2 = 2 * ones(n,1);

        X = [X1 ; X2];
        Y = [Y1 ; Y2];

    case 'swiss'

        tt = (3*pi/2) * (1+2*rand(1,n));
        X1 = [tt.*cos(tt) ; tt.*sin(tt)]';

        tt = (3*pi/2) * (1+2*rand(1,n));
        X2 = 1.6 * ( [tt.*cos(tt) ;  tt.*sin(tt)]' );
        X = [X1 ; X2] / 25 + randn(2*n,2) * 0.03;
        Y = [ones(n,1) ; 2 * ones(n,1)];

    otherwise
        error(['Unknown method ' method])
end
