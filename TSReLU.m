function g = TSReLU(z, derivate)
%17, TSReLU

sigmoid = 1./(1+exp(-z));

 if ~exist('derivate','var')
     g = z .* tanh(sigmoid);
     return;
 end

if derivate ~= 0
    %urata derivata
    ts = tanh(sigmoid);
    g = ts + z.*(1-ts.^2) .* sigmoid .*(1-sigmoid);
end