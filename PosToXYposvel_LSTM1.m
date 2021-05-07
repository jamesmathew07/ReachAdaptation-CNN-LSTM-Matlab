clc;clear all;close all;
% load synthetic data
load('LQGxyBase.mat');
Base  = Traj;

%% Pos X-Y cartesian only straight dir.
% In = [0 0 0 0.16]; % Xstart, Ystart, Xend, Yend
for i = 1:size(Base.In,2)
tTra{i,1} = [Base.Out(1:360,i)];
xTra{i,1} = [Base.In(1:4,i)];
end

%% lstm
numFeatures = 4;
numResponses = 360;
numHiddenUnits = 500;

layers = [ ...
    sequenceInputLayer(numFeatures)
    bilstmLayer(500,'Name','bilstm1') %lstmLayer(numHiddenUnits,'OutputMode','sequence')
    fullyConnectedLayer(numResponses)
    regressionLayer];

options = trainingOptions('adam', ...
    'MaxEpochs',1000, ...
    'GradientThreshold',1, ...
    'InitialLearnRate',0.005, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropPeriod',125, ...
    'LearnRateDropFactor',0.2, ...
    'Verbose',0, ...
    'Plots','training-progress');

net = trainNetwork(xTra,tTra,layers,options);
%%
analyzeNetwork(net)
%% predict
for i = 1:8
YPred(:,i) = predict(net,xTra{i,1});
subplot(131);plot(YPred(1:60,i),YPred(61:120,i));hold on;
subplot(132);plot(YPred(121:180,i),YPred(181:240,i));hold on;
subplot(133);plot(YPred(241:300,i),YPred(301:360,i));hold on;
end

lstm1PosToCon  = net;
save lstm1PosToCon 