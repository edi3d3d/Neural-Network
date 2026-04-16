%rng(0);
% 1. Set Data 
raw = readtable("data.csv");

Data = table2array(raw(:, 1:36));

target = string(raw.Target);


% Keep only Dropout and Graduate
mask = target ~= "Enrolled";

Tickets = zeros(sum(mask), 1);

Data = Data(mask, :);
target = target(mask);

Tickets(target == "Dropout")  = 0;
Tickets(target == "Graduate") = 1;

% 2. 80-20%
[n, ~] = size(Data);
idx = randperm(n);
split = round(0.8 * n);

trainIdx = idx(1:split);
testIdx = idx(split+1:end);

DataTrain = Data(trainIdx, :);
DataTest = Data(testIdx, :);

TicketsTrain = Tickets(trainIdx, :);
TicketsTest = Tickets(testIdx, :);

% 3. Normalize based on train
ColMean = mean(DataTrain, 1);
ColStd = std(DataTrain, 0, 1);
ColStd(ColStd == 0) = 1;

DataTrain = (DataTrain - ColMean) ./ ColStd;
DataTest = (DataTest - ColMean) ./ ColStd;

% 4. Bias
DataTrain = [DataTrain, ones(size(DataTrain,1), 1)];
DataTest = [DataTest, ones(size(DataTest,1), 1)];


% 5. Network settings

Inputs = size(Data, 2); % 36
Hidden = 14;

networkSGD.W = 0.005 * randn(Inputs + 1, Hidden);
networkSGD.w = 0.005 * randn(Hidden, 1);

networkSGDM.W = 0.005 * randn(Inputs + 1, Hidden);
networkSGDM.w = 0.005 * randn(Hidden, 1);

networkADAM.W = 0.005 * randn(Inputs + 1, Hidden);
networkADAM.w = 0.005 * randn(Hidden, 1);

% 6.0. sgd, random batch
SGDsettings.lr = 1e-3;
SGDsettings.maxIter = 1e4;
SGDsettings.batchSize = 32;

tic 
resultsSGD = sgd(DataTrain, TicketsTrain, networkSGD, SGDsettings, 0.20);
toc  %time for training

% 6.1. Sgdm, random batch, momentum
SGDMsettings.lr = 1e-3;
SGDMsettings.maxIter = 1e4;
SGDMsettings.batchSize = 32;
SGDMsettings.momentum = 0.9;

tic 
resultsSGDM = sgdm(DataTrain, TicketsTrain, networkSGDM, SGDMsettings, 0.20);
toc  %time for training

% 6.2 ADAM, sgdm with adam
ADAMsettings.lr = 1e-3;
ADAMsettings.maxIter = 1e4;
ADAMsettings.batchSize = 32;
ADAMsettings.beta1 = 0.9;
ADAMsettings.beta2 = 0.999;

tic 
resultsADAM = adam(DataTrain, TicketsTrain, networkADAM, ADAMsettings, 0.20);
toc  %time for training

%% 7. Evaluation

maxThreshold = 0.5;

[outSGD, ~] = forwardProp(DataTest, resultsSGD.WFinal, resultsSGD.wFinal);
SGDscores = scores(TicketsTest, outSGD, maxThreshold);

[outSGDM, ~] = forwardProp(DataTest, resultsSGDM.WFinal, resultsSGDM.wFinal);
SGDMscores = scores(TicketsTest, outSGDM, maxThreshold);

[outADAM, ~] = forwardProp(DataTest, resultsADAM.WFinal, resultsADAM.wFinal);
ADAMscores = scores(TicketsTest, outADAM, maxThreshold);

% Compute means
meanRecall = mean([SGDscores.Recall, SGDMscores.Recall, ADAMscores.Recall]);
meanPrecision = mean([SGDscores.Precision, SGDMscores.Precision, ADAMscores.Precision]);
meanSpecificity = mean([SGDscores.Specificity, SGDMscores.Specificity, ADAMscores.Specificity]);
meanAccuracy = mean([SGDscores.Accuracy, SGDMscores.Accuracy, ADAMscores.Accuracy]);
meanF1 = mean([SGDscores.F1, SGDMscores.F1, ADAMscores.F1]);

% Print table
fprintf('\n%-12s | %-9s | %-9s | %-11s | %-9s | %-9s\n', ...
    'Model', 'Recall', 'Precision', 'Specificity', 'Accuracy', 'F1');
fprintf('%s\n', repmat('-',1,75));

fprintf('%-12s | %-9.4f | %-9.4f | %-11.4f | %-9.4f | %-9.4f\n', ...
    'SGDscores', SGDscores.Recall, SGDscores.Precision, SGDscores.Specificity, SGDscores.Accuracy, SGDscores.F1);

fprintf('%-12s | %-9.4f | %-9.4f | %-11.4f | %-9.4f | %-9.4f\n', ...
    'SGDMscores', SGDMscores.Recall, SGDMscores.Precision, SGDMscores.Specificity, SGDMscores.Accuracy, SGDMscores.F1);

fprintf('%-12s | %-9.4f | %-9.4f | %-11.4f | %-9.4f | %-9.4f\n', ...
    'ADAMscores', ADAMscores.Recall, ADAMscores.Precision, ADAMscores.Specificity, ADAMscores.Accuracy, ADAMscores.F1);

fprintf('%s\n', repmat('-',1,75));

fprintf('%-12s | %-9.4f | %-9.4f | %-11.4f | %-9.4f | %-9.4f\n', ...
    'Mean', meanRecall, meanPrecision, meanSpecificity, meanAccuracy, meanF1);

% 8. Graphs

    % 8.0 SGD
    plotEvo(resultsSGD, [1 0 0]);
    hold on
    % 8.1 SGDM
    plotEvo(resultsSGDM, [0 0.5 0]);

    % 8.2 ADAM
    plotEvo(resultsADAM, [0 0 1]);
    hold off

    
    
    return ;
%% 9. Confusion matrix
    %% 9.0. sgd

    pred = double(outSGD >= 0.5);
    actual = double(TicketsTest);
    
    pred = categorical(pred, [0 1]);
    actual = categorical(actual, [0 1]);
    
    confusionchart(actual, pred);
    %% 9.1. sgdm

    pred = double(outSGDM >= 0.5);
    actual = double(TicketsTest);
    
    pred = categorical(pred, [0 1]);
    actual = categorical(actual, [0 1]);
    
    confusionchart(actual, pred);
    
    %% 9.2. adam
    
    pred = double(outADAM >= 0.5);
    actual = double(TicketsTest);
    
    pred = categorical(pred, [0 1]);
    actual = categorical(actual, [0 1]);
    
    confusionchart(actual, pred);