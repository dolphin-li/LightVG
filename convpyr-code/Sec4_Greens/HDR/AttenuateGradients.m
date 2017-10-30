function F = AttenuateGradients(I,alpha,beta)

% This is a loose interpretation of [Fattal et al. 2002] multi-scale 
% compression scheme for high-dynamic range images

[r c] = size(I);

maxDim = max(r, c);
levels = floor(log2(maxDim)) - log2(32) - 1;

Pyr = cell(levels,1);
Pyr{1} = I;

for i=2:levels

    Pyr{i} = imresize(Pyr{i-1} , 0.5, 'bilinear');
    
end

% Accumulate the attenuation
F = atten(I, 1);
for i=2:levels
    atten_i = imresize(atten(Pyr{i}, i), [r c], 'bilinear');
    F = F .* atten_i;
    
end

function res = atten(In, i) 

    dx = imfilter(In ,[-1 0 1]  ,'replicate')*1/2.^i;
    dy = imfilter(In ,[-1 0 1]'  ,'replicate')*1/2.^i;

    HM = sqrt(dx.^2 +dy.^2);

    % zero safety
    B = (HM == zeros(size(HM)));
    HM(B) = 1e-6;

    res = (alpha./HM);
    res = res.*((HM./alpha).^beta);
    res(HM<alpha) = 1; 


end

end
