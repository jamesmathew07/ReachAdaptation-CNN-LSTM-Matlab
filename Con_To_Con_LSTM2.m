clc;clear all;close all;
% load synthetic data
load('LQGxyBase.mat');
Base  = Traj;

%% visualise  decoded data
for i= 1
    p1 = Base.Out(1:60,i);
    p2 = Base.Out(61:120,i);
    v1 = Base.Out(121:180,i);
    v2 = Base.Out(181:240,i);
    c1 = Base.Out(241:300,i);
    c2 = Base.Out(301:360,i);
    
    subplot(231);plot(p1,p2,'k','Linewidth',2); title('Position Base');hold on; xlim([-0.04 0.04]);
    subplot(233);plot(c1,'k','Linewidth',2); title('Control Base');hold on; ylim([-30 30]);
    subplot(235);plot(v1,v2,'k','Linewidth',2); title('Velocity Base');hold on; xlim([-0.3 0.3]);
end

%% data reformating
% for lstm
for  i = 1:size(Base.Out,2)
    BaseCon2{i,1}  = Base.Out(241:360,i);
    BaseVel2{i,1}  = Base.Out(121:240,i);
    Con2{i,1}      = Base.Out(241:360,i);
 end
%% lstm2
numFeatures    = 120;
numResponses   = 120;

layers = [ ...
    sequenceInputLayer(numFeatures)
    bilstmLayer(500,'Name','bilstm1') %lstmLayer(numHiddenUnits,'OutputMode','sequence')
    dropoutLayer(0.2)
    bilstmLayer(100,'Name','bilstm2') %lstmLayer(numHiddenUnits,'OutputMode','sequence')
    dropoutLayer(0.2)
    fullyConnectedLayer(numResponses)
    regressionLayer];

options = trainingOptions('adam', ...
    'MaxEpochs',300, ...
    'GradientThreshold',1, ...
    'InitialLearnRate',0.005, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropPeriod',125, ...
    'LearnRateDropFactor',0.2, ...
    'Verbose',0, ...
    'Plots','training-progress');

net = trainNetwork(Con2,Con2,layers,options);
%%
figure

YPred1   = predict(net,Con2{1});
subplot(251);
plot(YPred1(1:60,1),'m','Linewidth',2);hold on;
plot(Con2{1}(1:60),'k','Linewidth',2);hold on;
ylim([-30 30]);
title(' Control- Base'); xlabel('TimeStep');ylabel('X');
subplot(256);plot(Con2{1}(1:60),'k','Linewidth',2);hold on;
ylim([-30 30]);
title('Control - Base');xlabel('TimeStep');ylabel('X');

%% X-Y graph
figure

subplot(251);
plot(YPred1(1:60,1),YPred1(61:120,1),'m','Linewidth',2);hold on;plot(BaseCon2{1}(1:60),BaseCon2{1}(61:120),'k','Linewidth',2);hold on;
xlim([-20 20]);title('Control Signal - Base'); xlabel('X');ylabel('Y');
plot(Con2{1}(1:60),Con2{1}(61:120),'k','Linewidth',2);hold on;
xlim([-20 20]);title('Control Signal - Base'); xlabel('X');ylabel('Y');
subplot(256);
plot(Con2{1}(1:60),Con2{1}(61:120),'k','Linewidth',2);hold on;
xlim([-20 20]); title('Control - Base'); xlabel('X');ylabel('Y');


%% 
lstm2ConToCon = net;
save('lstm2ConToCon.mat','lstm2ConToCon');