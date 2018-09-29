% import relevant files
myinputs = csvread('data146703.csv');
testinputs = csvread('data146703.csv');
% Reduce dimensionality of data as described in my report
myinputs = myinputs(:,[1,3]);
testinputs = testinputs(:,[1,3]);

% import precalculated weights
W = importdata('weights.mat');

% calculate distance between the training inputs and the inputs to be
% tested.
dist = pdist2(testinputs,myinputs);

% initialise output array
output=zeros(1,length(testinputs));

% itterate through inputs and produce an output for each
for i = 1:length(testinputs)
    for j = 1:myinputs
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        output(i)=output(i)+W(j)*dist(i,j);
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    end
end

% ouput is the completed matrix containing predictions based on the input