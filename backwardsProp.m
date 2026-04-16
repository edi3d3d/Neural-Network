function [gW, gw] = backwardsProp(In, tickets, w, forward)

[N, ~] = size(In);

err = forward.O - tickets;  %eroarea iesirii fata de iesirea corecta

gw = (forward.HO' * err) / N;  %gradientul pentru greutatile ultimului strat ascuns


HDO = TSReLU(forward.HL, 1);  %HIdden layer derivate output

deltaW = (err * w') .* HDO; %eroarea pe layer ascuns


gW = (In' * deltaW) / N;    %gradientul pentru greutatile de la intrare
end