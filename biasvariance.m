x1 = 0;   % lower interval border
x2 = 1;   % upper interval border
S = 100;  % number of data sets
N = 100;  % number of data pairs in each data set
K = 10;   % size of training set
sig = 0.2;  % mixing probability of error distributions
sigma0 = 0.1;   % standard deviation of first error distribution
sigma1 = 0.05;  % standard deviation of second error distribution
D = 6;  % D-1 highest degree of fitted polynomial
pt = [1 -0.5];   % target polynomial
pt = [zeros(1,D - length(pt)) pt]; % pad coefficient array with zeros
t = poly2sym(pt);
t = matlabFunction(t);

% for each data set D polynomials of degress 0 to D-1 are generated 
% with at most D coefficients
p = zeros(S,D,D);
% D average polynomials of degrees 0 to D-1 are calculated 
% with at most D coefficients
pav = zeros(D,D);
% squared training error
trainerror = zeros(S,K,D);
testerror = zeros(S,N-K,D);
bias = zeros(1,D);      % integrated squared bias
variance = zeros(1,D);  % integrated variance

for s = 1:S
    % generate data
    [x,y] = noisydata(t,x1,x2,N,sig,sigma0,sigma1);
    % choose training set
    trainindex = randperm(N,K);
    trainx = x(trainindex);
    trainy = y(trainindex);
    % let other data be test set
    testx = x;
    testx(trainindex) = [];
    testy = y;
    testy(trainindex) = [];

    for d = 1:D
        % returns the coefficients for a polynomial p0 of degree i-1 
        % that is a best fit (in a least-squares sense) for the data
        p0 = polyfit(trainx,trainy,d-1);
        p(s,d,:) = [zeros(1,D-d) p0];  % pad coefficient arays with zeros
        pav(d,:) = pav(d,:) + [zeros(1,D-d) p0];
        trainerror(s,:,d) = (trainy - polyval(p0, trainx)).^2;
        testerror(s,:,d) = (testy - polyval(p0, testx)).^2;
    end
end
pav = pav/S;

trainmean = mean(reshape(trainerror, [S*K,D]));
trainvar = var(reshape(trainerror, [S*K,D]));
testmean = mean(reshape(testerror, [S*(N-K),D]));
testvar = var(reshape(testerror, [S*(N-K),D]));
figure;
errorbar(trainmean,trainvar, 'Displayname', 'training error');
legend('show');
xticks(1:1:D);
xticklabels(0:1:D-1);
xlim([0 D+1]);figure;
errorbar(testmean,testvar, 'Displayname', 'test error');
legend('show');
xticks(1:1:D);
xticklabels(0:1:D-1);
xlim([0 D+1]);
figure;
plot(1:1:D, trainmean, 'Displayname', 'training error');
hold on;
plot(1:1:D, testmean, 'Displayname', 'testing error');
legend('show');
xticks(1:1:D);
xticklabels(0:1:D-1);
xlim([0 D+1]);

figure;
plot(x, t(x), 'k', 'DisplayName','truth');
hold on;
for d=1:D
    y1 = polyval(pav(d,:),x);
    plot(x,y1,'DisplayName',['degree ' num2str(d-1)]);
    f = pav(d,:) - pt;
    f = poly2sym(f);
    bias(d) = int(f^2,0,1);
    for s = 1:S
        p0 = p(s,d,:);
        f = p0 - pt;
        f = poly2sym(f);
        variance(d) = variance(d) + int(f^2,0,1);
    end
    variance = variance/S;
end
legend('show', 'Location', 'southeast');

figure;
plot(1:1:D, bias, 'Displayname', 'integrated squared bias');
legend('show');
xticks(1:1:D);
xticklabels(0:1:D-1);
xlim([0 D+1]);figure;
plot(1:1:D, variance, 'Displayname', 'integrated variance');
legend('show');
xticks(1:1:D);
xticklabels(0:1:D-1);
xlim([0 D+1]);