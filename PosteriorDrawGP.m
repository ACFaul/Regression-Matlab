test= -5:0.1:5;
test = test'; % Each row is a sample.
M = size(test,1);
% True function values.
t = 1 + test + sin(test);
% Noise variance.
noise = 0.00001;
% Number of training points.
N = 4;
train = -4:6/N:1;
train = train';
data = 1 + train + sin(train) + normrnd(0,sqrt(noise),N,1);
%kfcn = @(x,y,theta)  kInverseMQ(x,y,[theta(1), theta(2)]);
%theta0 = [2 3];
%theta = [1 1];
kfcn = @(x,y,theta)  kLinear(x,y,[theta(1), theta(2)]) ...
    + kGaussian(x,y,[theta(3), theta(4)]);
theta0 = [1 1 1 1];
theta = [1 1 1 1];
gprMdl = fitrgp(train,data,'KernelFunction',kfcn,'KernelParameters',theta0);
theta = [theta; ...
    gprMdl.KernelInformation.KernelParameters'];
for i=1:2 % draw first without, then with optimised parameters
k11 = kfcn(train,train,theta(i,:)) + noise*eye(N,N);
k21 = kfcn(test,train,theta(i,:));
k22 = kfcn(test,test,theta(i,:)) + noise*eye(M,M);
m = k21 * (k11\data);
k = k22 - k21* (k11\k21');
L = chol(k,'lower');
s = sqrt(diag(k));
figure;
plot(train, data, 'kd','MarkerFaceColor','k');
hold on;
plot(test, t, 'r-');
plot(test,m,'b-');
X = [test' fliplr(test')];
Y = [(m-2*s)' fliplr((m+2*s)')];
fill(X',Y', [0.7 0.7 0.7],'facea', 0.5,'edgecolor', 'none');
f1 = m+L*normrnd(0,1, M,1);
plot(test,f1,'k-');
f2 = m+L*normrnd(0,1, M,1);
plot(test,f2,'k--');
f3 = m+L*normrnd(0,1, M,1);
plot(test,f3,'k:');
f4 = m+L*normrnd(0,1, M,1);
plot(test,f4,'k-.');
legend('training data', 'true function', 'mean', ...
    '2 standard deviations', 'Location','northwest')
ylim([-5 6])
end