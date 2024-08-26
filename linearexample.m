% This script generates noisy data from a polynomial and splits this 
% into test and training data. It fits polynomials of various degrees 
% to it using least squares regression. The line, training and test
% data are plotted along with the fitted polynomials. The absolute
% training and test errors together with their variances are also
% plotted.

pt = [1 -0.5];   % target polynomial
t = poly2sym(pt);
t = matlabFunction(t);
x1 = 0;
x2 = 1;
N =100;   % number of data pairs
sig = 0.2;  % mixing probability of error distributions
sigma0 = 0.1;   % standard deviation of first error distribution
sigma1 = 0.05;  % standard deviation of second error distribution
% generate data
[x,y] = noisydata(t,x1,x2,N,sig,sigma0,sigma1);
% choose training set
K = 10;     % size of training set
trainindex = randi([1 N],1,K);
trainx = x(trainindex);
trainy = y(trainindex);
% let other data be test set
testx = x;
testx(trainindex) = [];
testy = y;
testy(trainindex) = [];

figure;
plot(x, t(x), 'k', 'DisplayName','truth');
hold on;
plot(trainx, trainy, 'ko', 'DisplayName','training data');
plot(testx, testy, 'k+', 'DisplayName','test data');

D = 6;  % D-1 highest degree of fitted polynomial
trainmean = zeros(1,D);
trainvar = zeros(1,D);
testmean = zeros(1,D);
testvar = zeros(1,D);
for d = 1:D
    p = polyfit(trainx,trainy,d-1);% returns the coefficients for a 
                                   % polynomial p of degree i-1 that
                                   % is a best fit (in a least-squares
                                   % sense) for the data
    trainerror = (trainy - polyval(p, trainx)).^2; % squared error
    trainmean(d) = mean(trainerror);
    trainvar(d) = var(trainerror);
    testerror = (testy - polyval(p, testx)).^2;
    testmean(d) = mean(testerror);
    testvar(d) = var(testerror);
    y1 = polyval(p,x);
    plot(x,y1,'DisplayName',['degree ' num2str(d-1)]);
end
legend('show', 'Location', 'southeast');
figure;
errorbar(trainmean,trainvar, 'Displayname', 'training error');
hold on;
errorbar(testmean,testvar, 'Displayname', 'test error');
legend('show');
xticks(1:1:D);
xticklabels(0:1:D-1);
xlim([0 D+1]);