function results = adam(data, tickets, network, settings, ts)

    if ~exist("ts",'var')
        ts = 0;
    end

    [N, ~] = size(data);

    W = network.W;
    w = network.w;

    % Moment1
    mW = zeros(size(W));
    mw = zeros(size(w));

    % RMS
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

        % First moment update
        mW = settings.beta1 * mW + (1 - settings.beta1) * gW;
        mw = settings.beta1 * mw + (1 - settings.beta1) * gw;

        % Second moment update
        vW = settings.beta2 * vW + (1 - settings.beta2) * (gW .^ 2);
        vw = settings.beta2 * vw + (1 - settings.beta2) * (gw .^ 2);

        % Bias correction
        mW_hat = mW / (1 - settings.beta1^iter);
        mw_hat = mw / (1 - settings.beta1^iter);

        vW_hat = vW / (1 - settings.beta2^iter);
        vw_hat = vw / (1 - settings.beta2^iter);

        % Weights updates
        W = W - settings.lr * mW_hat ./ (sqrt(vW_hat) + 1e-8);
        w = w - settings.lr * mw_hat ./ (sqrt(vw_hat) + 1e-8);


            %fprintf('Iteration: %d\n', iter);

            [yAll, ~] = forwardProp(data, W, w);

            % Prevent log(0)
            yAll = min(max(yAll, 1e-8), 1 - 1e-8);

            loss(iter) = -mean(tickets .* log(yAll) + (1 - tickets) .* log(1 - yAll));
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