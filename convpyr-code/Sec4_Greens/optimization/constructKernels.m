function [h1,h2,g] = constructKernels( w )
%CONSTRUCTKERNELS Construct the kernels from compact representation

h1 = w(1:3);
h1 = [h1 h1(end-1:-1:1)];
h1 = h1' * h1;

h2 = h1;

g = w(4:end);
g = [g g(end-1:-1:1)];
g = g' * g;

end

