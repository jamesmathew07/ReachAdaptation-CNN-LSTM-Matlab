clc;clear all;close all;
% load synthetic data
load('LQGxyBase.mat');
Base  = Traj;
load('LQGxy15CW.mat');
CW    = Traj;
load('LQGxy15CCW.mat');
CCW    = Traj;
load('LQGxy15CWadapt.mat');
CWa    = Traj;
load('LQGxy15CCWadapt.mat');
CCWa    = Traj;

%% visualise  decoded data
for i= 1:4:30%[1 4 7 10]%[12 10 8]
    p1 = Base.Out(1:60,i);
    p2 = Base.Out(61:120,i);
    v1 = Base.Out(121:180,i);
    v2 = Base.Out(181:240,i);
    c1 = Base.Out(241:300,i);
    c2 = Base.Out(301:360,i);
%     P1 = CCWa.Out(1:60,i);
%     P2 = CCWa.Out(61:120,i);
    P1 = CWa.Out(1:60,i);
    P2 = CWa.Out(61:120,i);
    V1 = CW.Out(121:180,i);
    V2 = CW.Out(181:240,i);   
    C1 = CW.Out(241:300,i);
    C2 = CW.Out(301:360,i);
    subplot(231);plot(p1,p2,'k','Linewidth',2); title('Position Base');hold on; xlim([-0.04 0.04]);
%     subplot(232);plot(P1,P2,'m','Linewidth',2); title('Position CW');hold on; xlim([-0.04 0.04]); ylim([0 0.2]);
    subplot(232);plot(P1,P2,'Linewidth',2); title('Position CW');hold on; xlim([-0.04 0.04]); ylim([0 0.2]);
    subplot(233);plot(c1,'k','Linewidth',2); title('Control Base');hold on; ylim([-30 30]);
    subplot(234);plot(C1,'r','Linewidth',2); title('Control CW');hold on; ylim([-30 30]);
    subplot(235);plot(v1,v2,'k','Linewidth',2); title('Velocity Base');hold on; xlim([-0.3 0.3]);
    subplot(236);plot(V1,V2,'r','Linewidth',2); title('Velocity CW');hold on; %xlim([-0.04 0.04
% pause
end

%% data reformating
% for lstm
j = 1;
for  i = 1:size(Base.Out,2)
    BaseCon2{i,1}  = Base.Out(241:360,i);
    BaseVel2{i,1}  = Base.Out(121:240,i);
    CWCon2{i,1}    = CW.Out(241:360,i);
    CWVel2{i,1}    = CW.Out(121:240,i);
    CCWCon2{i,1}   = CCW.Out(241:360,i);
    CCWVel2{i,1}   = CCW.Out(121:240,i);
    
    CWaCon2{i,1}   = CWa.Out(241:360,i);
    CWaVel2{i,1}   = CWa.Out(121:240,i);
    CCWaCon2{i,1}  = CCWa.Out(241:360,i);
    CCWaVel2{i,1}  = CCWa.Out(121:240,i);
    
    Con2{j,1}    = Base.Out(241:360,i);
    Con2{j+1,1}  = CW.Out(241:360,i);
    Con2{j+2,1}  = CWa.Out(241:360,i);
    Con2{j+3,1}  = CCW.Out(241:360,i);
    Con2{j+4,1}  = CCWa.Out(241:360,i);
    
    Vel2{j,1}    = Base.Out(121:240,i);
    Vel2{j+1,1}  = CW.Out(121:240,i);
    Vel2{j+2,1}  = CWa.Out(121:240,i);
    Vel2{j+3,1}  = CCW.Out(121:240,i);
    Vel2{j+4,1}  = CCWa.Out(121:240,i);
    
    j = j+5;
end
%% lstm4
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

net = trainNetwork(Vel2,Con2,layers,options);
%%
% analyzeNetwork(net)

%%
figure

YPred1   = predict(net,BaseVel2{1});
subplot(251);
plot(YPred1(1:60,1),'m','Linewidth',2);hold on;
plot(BaseCon2{1}(1:60),'k','Linewidth',2);hold on;
ylim([-30 30]);
title(' Control- Base'); xlabel('TimeStep');ylabel('X');
subplot(256);plot(BaseVel2{1}(1:60),'k','Linewidth',2);hold on;
ylim([-0.3 0.3]);
title('Velocity - Base');xlabel('TimeStep');ylabel('X');

YPred2   = predict(net,CWVel2{1});
subplot(252);
plot(YPred2(1:60,1),'m','Linewidth',2);hold on;
plot(CWCon2{1}(1:60),'k','Linewidth',2);hold on;
ylim([-30 30]);
title('ControlX - CW');
subplot(257);plot(CWVel2{1}(1:60),'k','Linewidth',2);hold on;
ylim([-0.3 0.3]);
title('Velocity- CW');

YPred3   = predict(net,CCWVel2{1});
subplot(253);
plot(YPred3(1:60,1),'m','Linewidth',2);hold on;
plot(CCWCon2{1}(1:60),'k','Linewidth',2);hold on;
ylim([-30 30]);
title('ControlX - CCW');
subplot(258);plot(CCWVel2{1}(1:60),'k','Linewidth',2);hold on;
ylim([-0.3 0.3]);
title('Velocity- CCW');

YPred4   = predict(net,CWaVel2{8});
subplot(254);
plot(YPred4(1:60,1),'m','Linewidth',2);hold on;
plot(CWaCon2{8}(1:60),'k','Linewidth',2);hold on;
ylim([-30 30]);
title('ControlX - CWadapt');
subplot(259);plot(CWaVel2{8}(1:60),'k','Linewidth',2);hold on;
ylim([-0.3 0.3]);
title('Velocity- CWadapt');

YPred5   = predict(net,CCWaVel2{8});
subplot(255);
plot(YPred5(1:60,1),'m','Linewidth',2);hold on;
plot(CCWaCon2{8}(1:60),'k','Linewidth',2);hold on;
ylim([-30 30]);
title('ControlX - CCWadapt');
subplot(2,5,10);plot(CCWaVel2{5}(1:60),'k','Linewidth',2);hold on;
ylim([-0.3 0.3]);
title(' Velocity- CCWadapt');

%% X-Y graph
figure

subplot(251);
plot(YPred1(1:60,1),YPred1(61:120,1),'m','Linewidth',2);hold on;plot(BaseCon2{1}(1:60),BaseCon2{1}(61:120),'k','Linewidth',2);hold on;
xlim([-20 20]);title('Control Signal - Base'); xlabel('X');ylabel('Y');
plot(BaseCon2{1}(1:60),BaseCon2{1}(61:120),'k','Linewidth',2);hold on;
xlim([-20 20]);title('Control Signal - Base'); xlabel('X');ylabel('Y');
subplot(256);
plot(BaseVel2{1}(1:60),BaseVel2{1}(61:120),'k','Linewidth',2);hold on;
xlim([-0.3 0.3]); title('Velocity - Base'); xlabel('X');ylabel('Y');


subplot(252);
plot(YPred2(1:60,1),YPred2(61:120,1),'m','Linewidth',2);hold on;
plot(CWCon2{1}(1:60),CWCon2{1}(61:120),'k','Linewidth',2);hold on;
title('Control - CW');% ylim([-30 30]);
subplot(257);
plot(CWVel2{1}(1:60),CWVel2{1}(61:120),'k','Linewidth',2);hold on;
xlim([-0.3 0.3]);title('Velocity - CW');

subplot(253);
plot(YPred3(1:60,1),YPred3(61:120,1),'m','Linewidth',2);hold on;
plot(CCWCon2{1}(1:60),CCWCon2{1}(61:120),'k','Linewidth',2);hold on;
title('Control - CCW'); ylim([-15 20]);
subplot(258);
plot(CCWVel2{1}(1:60),CCWVel2{1}(61:120),'k','Linewidth',2);hold on;
xlim([-0.3 0.3]);title('Velocity - CCW');

subplot(254);
plot(YPred4(1:60,1),YPred4(61:120,1),'m','Linewidth',2);hold on;
plot(CWaCon2{5}(1:60),CWaCon2{5}(61:120),'k','Linewidth',2);hold on;
title('Control - CW adapt'); ylim([-15 20]);
subplot(259);
plot(CWaVel2{5}(1:60),CWaVel2{5}(61:120),'k','Linewidth',2);hold on;
xlim([-0.3 0.3]);title('Velocity - CW adapt');

subplot(255);
plot(YPred5(1:60,1),YPred5(61:120,1),'m','Linewidth',2);hold on;
plot(CCWaCon2{5}(1:60),CCWaCon2{5}(61:120),'k','Linewidth',2);hold on;
title('Control - CCW adapt');% ylim([-30 30]);
subplot(2,5,10);
plot(CCWaVel2{5}(1:60),CCWaVel2{5}(61:120),'k','Linewidth',2);hold on;
xlim([-0.3 0.3]);title('Velocity - CCW adapt');
%% 
lstm4VelCon = net;
save('lstm4VelCon.mat','lstm4VelCon');

%% for ppt visualisation only
for i = [1 5 8 12]
YPred4   = predict(net,CWaVel2{i});
subplot(254);
% plot(YPred4(1:60,1),'m','Linewidth',2);hold on;
plot(CWaCon2{i}(1:60),'r','Linewidth',2);hold on;
ylim([-30 30]);
title('ControlX - CWadapt');
subplot(259);plot(CWaVel2{i}(1:60),'r','Linewidth',2);hold on;
ylim([-0.3 0.3]);
title('Velocity- CWadapt');

YPred5   = predict(net,CCWaVel2{i}); 
subplot(255);
% plot(YPred5(1:60,1),'m','Linewidth',2);hold on;
plot(CCWaCon2{i}(1:60),'r','Linewidth',2);hold on;
ylim([-30 30]);
title('ControlX - CCWadapt');
subplot(2,5,10);plot(CCWaVel2{i}(1:60),'r','Linewidth',2);hold on;
ylim([-0.3 0.3]);
title(' Velocity- CCWadapt');
pause
end