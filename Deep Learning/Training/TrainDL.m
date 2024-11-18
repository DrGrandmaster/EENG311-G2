%% Creates a Neural Network to Forcast the Next Element of a Time Series

% Configuration
trainingDataFilename = "sequence_DIAtemp_test.mat";
exportFilename = "DIA_Model";

% Network Layers
layers = [
    sequenceInputLayer(1,"Name","sequenceinput")
    fullyConnectedLayer(100,"Name","fc_1")
    selfAttentionLayer(4,256,"Name","selfattention")
    fullyConnectedLayer(100,"Name","fc_2")
    geluLayer("Name","gelu")
    fullyConnectedLayer(100,"Name","fc_3")
    geluLayer("Name","gelu_1")
    lstmLayer(128,"Name","lstm","OutputMode","last")
    fullyConnectedLayer(9,"Name","fc")
    softmaxLayer("Name","softmax")
    classificationLayer("Name","classoutput")];

% Load Training Data
load(trainingDataFilename);

% Prepare Data to Train
% Slice sequence so the model learns to predict the next symbol
XTrain{numel(sequence)-1, 1} = [];
for i = 1:numel(sequence)-1

    XTrain{i} = sequence(1:i)';

end
YTrain = categorical(sequence(2:end));

% Configure Training Options
opts = trainingOptions( ...
    "adam", ...
    "InitialLearnRate", 0.01, ...
    "LearnRateSchedule","piecewise", ...
    "LearnRateDropPeriod", 2, ...
    MaxEpochs=100, ...
    SequencePaddingDirection="left", ...
    Plots="training-progress", ...
    Verbose=0 ...
);

% Train Network
net = trainNetwork(XTrain, YTrain, layers, opts);

disp('Network Trained.');
disp('Press any key to continue.');
pause;

% Export Network to .mat File
save(exportFilename, "net");