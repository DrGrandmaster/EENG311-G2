function [symbol,penaltyInBits] = symbolMachineF24(probs)
% function [symbol,penaltyInBits] = symbolMachineF24(probs)
%
% Runs one step of the Symbol Machine. After initializing the Symbol
% Machine with initializeSymbolMachineF24.m, symbolMachineF24.m should be
% called once for each symbol appearing in the sequence.
%
% Inputs:
%   probs: 9x1 or 1x9 vector of probabilities forecasting the next symbol
%       to appear in the sequence.
%
% Outputs:
%   symbol: the next symbol that appears in the sequence
%   penaltyInBits: the penalty for the forecast, measured as the negative
%       log (base 2) of the forecasted probability of the symbol that appears
%
% Note: initializeSymbolMachineF24.m must be used before the first call to
%     symbolMachineF24.m.
%
% Colorado School of Mines EENG311 - Fall 2024 - Mike Wakin

global SYMBOLDATA

% Make sure probs are nonnegative
if min(probs) < 0
    error('Vector probs cannot contain negative numbers.');
end

% Make sure probs sum to 1
if abs(sum(probs)-1) > 1e-6
    error('Vector probs does not sum to 1.');
end

if strcmp(SYMBOLDATA.initializerVersion,'F24')
    SYMBOLDATA.machineVersion = 'F24';
else
    error('Symbol Machine should first be initialized with initializeSymbolMachineF24.m.');
end


if SYMBOLDATA.nextIndex <= SYMBOLDATA.sequenceLength
    % Extract next symbol in sequence
    symbol = SYMBOLDATA.sequence(SYMBOLDATA.nextIndex);
    % Assign penalty based on forecasted probability of this symbol
    penaltyInBits = -log2(probs(symbol));
    SYMBOLDATA.totalPenaltyInBits = SYMBOLDATA.totalPenaltyInBits + penaltyInBits;
    % Check if this symbol was forecasted as the most likely (in case of
    % tie, choose one randomly from those forecasted as most likely)
    maxIndices = find(probs==max(probs));
    if symbol==maxIndices(randi(length(maxIndices)))
        SYMBOLDATA.correctPredictions = SYMBOLDATA.correctPredictions + 1;
    end
    % Keep track of the forecasted probability of the appearing symbols
    SYMBOLDATA.winnerProbabilities(SYMBOLDATA.nextIndex) = probs(symbol);
    % Keep track of the forecasted probabilities of the non-appearing symbols
    SYMBOLDATA.loserProbabilities(SYMBOLDATA.nextIndex,:) = probs(setdiff(1:9,symbol));
    % Keep track of all forecasted probabilities
    SYMBOLDATA.forecastedProbabilities(SYMBOLDATA.nextIndex,:) = probs; 
    % Increment index in symbol sequence
    SYMBOLDATA.nextIndex = SYMBOLDATA.nextIndex+1;
else
    error('No more symbols available. Call initializeSymbolMachineF24.m if needed.');
end

if SYMBOLDATA.verbose
    fprintf('Symbol %d of %d is %d, which was forecasted with probability %1.7f. Penalty is %4.3f bits.\n',SYMBOLDATA.nextIndex-1,SYMBOLDATA.sequenceLength,symbol,probs(symbol),penaltyInBits);
end
