x= (-5:0.1:5)'; % Each row is a sample.

k = kGaussian(x,x, [1 1]);
% Note that sometimes it is necessary to add 1e-15*eye(size(x,1)) 
% to k to ensure positive definiteness due to rounding errors.
L = chol(k,'lower');
figure;
f1 = L*normrnd(0,1, size(x,1),1);
plot(x,f1,'k-');
hold on;
f2 = L*normrnd(0,1, size(x,1),1);
plot(x,f2,'k--');
f3 = L*normrnd(0,1, size(x,1),1);
plot(x,f3,'k:');
f4 = L*normrnd(0,1, size(x,1),1);
plot(x,f4,'k-.');

% Each row of x and y is one data sample. The functions below
% calculate the covariance matrix for a specific kernel.

% Constant kernel.
function k = kConst(x,y, param)
k =param^2* ones(size(x,1), size(y,1));
end

% Linear kernel.
function k = kLinear(x,y, params)
% The matrix of the inner products of each row of x and each row of y 
% is given by x*y'.
k = params(1)^2 + params(2)^2*x*y';
end

% Quadratic kernel.
function k = kQuadratic(x,y, params)
k = (params(1)^2 + params(2)^2*x*y').^2;
end

% Gaussian kernel.
function k = kGaussian(x,y, params)
% Calculate squared distance as the sum of the inner product of one
% row of x with itself and one row of y with itself minus twice the
% inner product of these two rows.
sd = repmat(dot(x,x,2),1,size(y,1)) + ...
    repmat(dot(y,y,2)',size(x,1),1) - 2*x*y';  
k = params(1)^2* exp(-sd/(params(2)^2*2));
end

% Exponential kernel.
function k = kExponential(x,y,params)
sd = repmat(dot(x,x,2),1,size(y,1)) + ...
    repmat(dot(y,y,2)',size(x,1),1) - 2*x*y';  
k = params(1)^2 * exp(-sqrt(sd)/params(2));
end

% Inverse multiquadric kernel.
function k = kInverseMQ(x,y, params)
sd = repmat(dot(x,x,2),1,size(y,1)) + ...
    repmat(dot(y,y,2)',size(x,1),1) - 2*x*y';
k = params(1)^2./sqrt(1+sd/(params(2)*2));
end