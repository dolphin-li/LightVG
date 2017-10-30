function d = optHandle( w )
%OPTHANDLE Evaluate & compute the error

global cI;
global divG;

[h1,h2,g] = constructKernels( w );

% Evaluate
res = evalf( -divG, h1, h2, g );    

% Error
d = abs(cI-res);
d = mean(d(:))

end

