function results = sgd(data, tickets, network, settings, ts)
    
    if ~exist("ts",'var')
        ts = 0;
    end
    [N, ~] = size(data);
    
    W = network.W;
    w = network.w;
    
    loss = zeros(settings.maxIter, 1);
    grad = zeros(settings.maxIter, 1);
    
    for iter = 1:settings.maxIter
        randomBatch = randperm(N, settings.batchSize);
        In1 = data(randomBatch, :);
        Tickets1 = tickets(randomBatch);
    
        [~, forward] = forwardProp(In1, W, w);
        [gW, gw] = backwardsProp(In1, Tickets1, w, forward);
    
        W = W - settings.lr * gW;
        w = w - settings.lr * gw;
    
        
            %fprintf('Iteration: %d\n', iter);
            [yAll, ~] = forwardProp(data, W, w);
    
            %yAll can be <0 and with a log(yAll) it gives a complex part that
            %doesnt like to be plotted, so i capped it
    
            yAll = min(max(yAll, 1e-8), 1 - 1e-8);
    
            loss(iter) = -mean(tickets .* log(yAll) + (1 - tickets) .* log(1-yAll));          %rewrite
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