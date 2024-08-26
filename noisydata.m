function [x,y] = noisydata(f,x1,x2,N,s,sigma0,sigma1)
% This function creates N noisy data pairs (x,y) from a function f
% in the interval [x1,x2] with added noise where the noise is a 
% mixture of zero mean normal distributions with variances sigma0
% and sigma1 with probability s and 1-s.
% Input arguments:
%   f,      function handle
%   x1,x2,  end points of interval
%   N,      number of data pairs to be generated
%   s,      mixing probability of error distributions
%   sigma0, standard deviation of first error distribution
%   sigma1, standard of second error distribution
x = linspace(x1,x2,N);
y = zeros(1,N);
for i = 1:N
	r = rand;
	if (r<=s)
      y(i) = f(x(i)) + randn*sigma1;
    else
      y(i) = f(x(i)) + randn*sigma0;
    end
end
