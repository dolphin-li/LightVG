function d = optHandle( w )
%OPTHANDLE Evaluate & compute the error

global I;
global rhs;

% Evaluate
res = evalg( rhs, w );    

% Error
d = (I-res).^2;
d = mean(d(:));

end

