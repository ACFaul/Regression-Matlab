pt = [1 -0.5];   % target polynomial
t = poly2sym(pt);
t = matlabFunction(t);
x1 = 0;
x2 = 1;
N =100;   % number of data pairs
x = linspace(x1,x2,N);
figure;
plot(x, t(x), 'k', 'DisplayName','truth');
hold on;

sig = 0.2;  % mixing probability of error distributions
sigma0 = 0.1;   % standard deviation of first error distribution
sigma1 = 0.05;  % standard deviation of second error distribution
D = 6;  % D-1 highest degree of fitted polynomial
P = zeros(6,6);  % accumulate coefficients
K = 20;     % size of training set
% generate data
for a = 1:100
    [x,y] = noisydata(t,x1,x2,N,sig,sigma0,sigma1);
    % choose training set
    trainindex = randi([1 N],1,K);
    trainx = x(trainindex);
    trainy = y(trainindex);
    for d = 1:D
        % returns the coefficients for a polynomial p of degree i-1 that
        % is a best fit (in a least-squares sense) for the data
        P(d,:) = P(d,:) + cat(2,zeros(1,D-d),polyfit(trainx,trainy,d-1));
    end
end
P = P./100;
for d = 1:D
    y1 = polyval(P(d,:),x);
    plot(x,y1,'DisplayName',['degree ' num2str(d-1)]);
end
legend('show', 'Location', 'north');

