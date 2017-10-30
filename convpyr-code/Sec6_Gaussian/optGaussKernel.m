function w = optGaussKernel( sigma )
% OPTGAUSSKERNEL Optimize set F for gaussian with a paramter
% sigma

global I;
global rhs;

% Convolution input - just delta.
n = 128;
rhs = zeros(n,n);
rhs(n/2, n/2) = 1;        

% Convolution output - gaussian.
[X,Y] = meshgrid(-n:n, -n:n);  
h = exp(-(X.^2 + Y.^2)/(2*sigma^2));
I = imfilter(rhs, h,0); 

% Optimize for w.
% Unsatisfactory results -> More iterations. 
% See: optimset in Matlab documentation.
w0 = rand(12,1);
w = fminunc(@optHandle,double(w0));

end

