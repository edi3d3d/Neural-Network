function [Out, forward] = forwardProp(In, W, w)

HiddenLayer = In * W; %Hidden layer values with the weighted input
HiddenOutputs = TSReLU(HiddenLayer);  %HIdden layer output 

OutputNeuron = HiddenOutputs * w;

Out = 1./(1+exp(-OutputNeuron));

forward.HL = HiddenLayer;
forward.HO = HiddenOutputs;
forward.ON = OutputNeuron;
forward.O = Out;

end