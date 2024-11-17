function sequenceLength = initializeSymbolMachineF24(filename,verbose)
% function sequenceLength = initializeSymbolMachineF24(filename,verbose)
%
% Initializes global variables used by the Symbol Machine. 
%
% Inputs:
%   filename: name of .mat file (such as 'dataset.mat', including quotes) 
%       containing one sequence of symbols stored in a vector called 
%       sequence. Each symbol in the sequence must be a digit from 1 to 9.
%   verbose (optional): if true (or 1), Symbol Machine will output each
%       symbol and its penalty. default = false.
% 
% Outputs:
%   sequenceLength: the total number of symbols in the sequence
% 
% Colorado School of Mines EENG311 - Fall 2024 - Mike Wakin

clear global SYMBOLDATA
global SYMBOLDATA

eval(['load ' filename]);

sequenceLength = length(sequence);

SYMBOLDATA.filename = filename;
SYMBOLDATA.sequence = sequence;
SYMBOLDATA.sequenceLength = sequenceLength;
SYMBOLDATA.nextIndex = 1;
SYMBOLDATA.totalPenaltyInBits = 0;
SYMBOLDATA.correctPredictions = 0;
SYMBOLDATA.winnerProbabilities = zeros(sequenceLength,1);
SYMBOLDATA.loserProbabilities = zeros(sequenceLength,8);
SYMBOLDATA.forecastedProbabilities = zeros(sequenceLength,9);
SYMBOLDATA.initializerVersion = 'F24';

if nargin == 2
    if verbose
        SYMBOLDATA.verbose = true;
    else
        SYMBOLDATA.verbose = false;
    end
else
    SYMBOLDATA.verbose = false;
end

fprintf('Initialized Symbol Machine for %s, which contains %d symbols.\nUse symbolMachineF24.m to make forecasts.\n\n',SYMBOLDATA.filename,SYMBOLDATA.sequenceLength);
