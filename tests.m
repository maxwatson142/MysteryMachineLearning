%% fidling with data
inputs = csvread('data146703.csv');
outputs = csvread('testresults146703.csv');
%reduce dimensions
inputs = inputs(:,[1,3]);
%whiten
%[inputs,mu,invMat] = whiten(inputs);

% split data into training and testing
ttInputs = mat2cell(inputs,[100,100,100,100,100,100,100,100,100,100],2);
ttOutputs = mat2cell(outputs,[100,100,100,100,100,100,100,100,100,100],1);
trainIn = cell2mat([ttInputs(1,1);ttInputs(2,1);ttInputs(3,1);ttInputs(4,1);ttInputs(5,1);ttInputs(7,1);ttInputs(8,1);ttInputs(9,1);ttInputs(6,1)]);
trainOut = cell2mat([ttOutputs(1,1);ttOutputs(2,1);ttOutputs(3,1);ttOutputs(4,1);ttOutputs(5,1);ttOutputs(7,1);ttOutputs(8,1);ttOutputs(9,1);ttOutputs(6,1)]);
testIn = cell2mat(ttInputs(10,1));
testOut = cell2mat(ttOutputs(10,1));
trainSize = 900 ;
%% Exact Identity RBF
G = zeros(trainSize,trainSize);
G = pdist2(trainIn,trainIn);

% calculate weights
W = G\trainOut;

%iterate through testing data and produce outputs
dist = pdist2(testIn,trainIn);
output=zeros(1,length(testIn));
for i = 1:length(testIn)
    for j = 1:trainSize
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        output(i)=output(i)+W(j)*dist(i,j);
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    end
end

%calculate mean square loss
means = testOut - output';
total = 0;
for i = 1:length(means)
    total = total + means(i)^2;
end

mslEI = total/length(means);


%% Exact Gausian
sigma = 0.001;
G = zeros(trainSize,trainSize);
dist= pdist2(trainIn,trainIn);
for i = 1:trainSize
    for j = 1:trainSize
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        G(i,j) = exp(-(((dist(i,j))^2)/(2*sigma^2)));
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    end
end
 
% calculate weights
W = inv(G)\trainOut;

%iterate through testing data and produce outputs
output=zeros(1,length(testIn));
for i = 1:length(testIn)
    for j = 1:trainSize
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        d = exp(-(((dist(i,j))^2)/(2*sigma^2)));
        output(i)=output(i)+W(j)*d; 
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    end
end

%calculate mean square loss
means = testOut - output';
total = 0;
for i = 1:length(means)
    total = total + means(i)^2;
end

mslEG = total/length(means);

%% Clustering

k=12;

[clustering,C] = kmeans(trainIn,k);

%% Identity RBFN

G = zeros(k,trainSize);
dist= pdist2(trainIn,C);
for i = 1:k
    for j = 1:trainSize
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        G(i,j) = dist(j,i);
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    end
end
% calculate weights
W = pinv(G')*trainOut;
dist= pdist2(testIn,C);
%iterate through testing data and produce outputs
for i = 1:length(testIn)
    for j = 1:k
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        d= dist(i,j);
        output(i)=output(i)+W(j)*d;% to complete
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    end
end

%calculate mean square loss
means = testOut - output';
total = 0;
for i = 1:length(means)
    total = total + means(i)^2;
end

mslIRBFN = total/length(means);

%% gausian RBFN

sigma = 0.3;

G = zeros(k,trainSize);

dist= pdist2(trainIn,C);

for i = 1:k
    for j = 1:trainSize
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
         G(i,j) = exp(-(((dist(j,i))^2)/(2*sigma^2)));
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    end
end
% calculate weights
W = pinv(G')*trainOut;

%iterate through testing data and produce outputs
for i = 1:length(testIn)
    for j = 1:k
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        d = exp(-(((dist(i,j))^2)/(2*sigma^2)));
        output(i)=output(i)+W(j)*d;
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    end
end

%calculate mean square loss
means = testOut - output';
total = 0;
for i = 1:length(means)
    total = total + means(i)^2;
end

mslGRBFB = total/length(means);