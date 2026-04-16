function results = sgdm(data, tickets, network, settings, ts)
    
    if ~exist("ts",'var')
        ts = 0;
    end
    
    [N, ~] = size(data);
    
    W = network.W;
    w = network.w;
    
    vW = zeros(size(W));
    vw = zeros(size(w));
    
    loss = zeros(settings.maxIter, 1);
    grad = zeros(settings.maxIter, 1);
    
    for iter = 1:settings.maxIter
        randomBatch = randperm(N, settings.batchSize);
        In1 = data(randomBatch, :);
        Tickets1 = tickets(randomBatch);
    
        [~, forward] = forwardProp(In1, W, w);
        [gW, gw] = backwardsProp(In1, Tickets1, w, forward);
    
        %velocity update
    
        vW = settings.momentum * vW + settings.lr * gW;
        vw = settings.momentum * vw + settings.lr * gw;
    
        W = W - vW;
        w = w - vw;
    
    
        
            %fprintf('Iteration: %d\n', iter);
            [yAll, ~] = forwardProp(data, W, w);
    
            %yAll can be <0 and with a log(yAll) it gives a complex part that
            %doesnt like to be plotted, so i capped it
    
            %fixed the yAll being <0, the output activation function was not
            %sigmoid so the output was not a probability
    
            yAll = min(max(yAll, 1e-8), 1 - 1e-8);
    
            loss(iter) = -mean(tickets .* log(yAll) + (1 - tickets) .* log(1-yAll));
            grad(iter) = sqrt(sum(gW(:).^2) + sum(gw(:).^2));
    
            if loss(iter) < ts
                loss = loss(1:(iter));
                grad = grad(1:(iter));
                break;
            end
    end
    results.loss = loss;
    results.grad = grad;

    results.WFinal = W;
    results.wFinal = w;
end