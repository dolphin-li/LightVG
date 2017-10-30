function res = evalg( x, w )
% EVALG - evaluate convolution pyramid on input vector x
% in a gaussian case.

% For the details on why gaussian case needs a special treatment
% in our framework and for the reason that we need to introduce 
% per-level scaling factors please have a look in the paper.

% We could optimize for h1/h2 as well, but it will
% make the convergence much slower. Using Burt's
% kernels works well in a gaussian case.
h1 = [1 4 6 4 1]/16.0;
h1 = h1' * h1;

% Instead we optimize for: 
% w(1)    -  Scaling parameter of the upsampling kernel
% w(2:5) -  Four DOFs of symmetric, separable 7x7 filter 
% w(6:end)  - Scaling parameter for each level

h2 = h1*w(1); 

g = w(2:5)';
g = [g g(end-1:-1:1)];
g = g' * g;

sl = 6; % first scaling inex
fs = 5; % boundary padding value
maxLevel = log2(size(x,1));

% Analysis
pyr{1} = padarray(x, [fs fs],'replicate');
for i=2:maxLevel
    
    down = imfilter(pyr{i-1},h1,0);
    down = down(1:2:end,1:2:end);
    
    down = padarray(down, [fs fs],'replicate');
    pyr{i} = down;
    
end

% Synthesis
fpyr{maxLevel} = imfilter(pyr{maxLevel}, g*w(sl),0);
for i=maxLevel-1:-1:1
    
    rd = fpyr{i+1};
    rd = rd(1+fs:end-fs, 1+fs:end-fs);
    
    up = zeros(size(pyr{i}));
    up(1:2:end,1:2:end) = rd;
    
    fpyr{i} = imfilter(up,h2,0) + imfilter(pyr{i}, g*w(sl+maxLevel-i),0);

end

res = fpyr{1};
res = res(1+fs:end-fs, 1+fs:end-fs);

end

