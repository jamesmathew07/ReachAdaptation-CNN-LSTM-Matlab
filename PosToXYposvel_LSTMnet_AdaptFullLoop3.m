clc;clear all;close all;
% load data and neural nets
load('LQGxy15CW.mat');
CW    = Traj;
load('LQGxy15CWadapt.mat');
CWa    = Traj;
load('LQGxy15CCW.mat');
CCW    = Traj;
load('LQGxyBase.mat');
Base  = Traj;
load lstm3ConVel.mat
load lstm4VelCon.mat
load lstm1PosToCon.mat
load lstm2ConToCon.mat

%% lstm1
net = lstm1PosToCon;
YPred = predict(net,[0 0 0 0.16]');
subplot(131);plot(YPred(1:60),YPred(61:120));hold on;xlim([-0.03 0.03]);title('Position'); xlabel('Y');ylabel('X');
subplot(132);plot(YPred(121:180),YPred(181:240));hold on;xlim([-0.3 0.3]);title('Velocity');
subplot(133);plot(YPred(241:300),YPred(301:360));hold on;xlim([-30 30]);title('Control Signal');

%% lstm2
in2    = YPred(241:360);
net    = lstm2ConToCon;
YPred2 = predict(net,in2);
figure(2);
subplot(131);plot(YPred2(1:60),YPred2(61:120));hold on;xlim([-30 30]);title('Control X vs Y'); xlabel('Y');ylabel('X');
subplot(132);plot(YPred2);hold on;title('control');xlabel('Time steps for X and y vectors');ylabel('control');
subplot(133);plot(YPred2(1:60));hold on;title('control X');xlabel('Time steps for X ');ylabel('control');
%% lstm3
net    = lstm3ConVel;
YPred3 = predict(net,YPred2);
figure(3);subplot(131);plot(YPred3(1:60),YPred3(61:120));hold on;xlim([-0.3 0.3]);title('Velocity');xlabel('Y');ylabel('X');
subplot(131);plot(YPred3);hold on;xlim([-0.3 0.3]);title('Velocity for X & Y');xlabel('vel');ylabel('Time steps X and then Y');
subplot(131);plot(YPred3(1:60));hold on;xlim([-0.3 0.3]);title('Velocity X');xlabel('Vel');ylabel('Timesteps');

%%
lstm2ConToConCopy2 = lstm2ConToCon;
save('lstm2ConToConCopy2.mat','lstm2ConToConCopy2');
%% perturbation +lstm4

YPred5 = YPred2;
kk=1;
for j = [1:4:30]
VelAct1 = CWa.Out(121:240,j);
net     = lstm4VelCon;
YPred4  = predict(net,VelAct1);
% figure(4);
% subplot(131);plot(YPred4(1:60),YPred4(61:120));hold on;xlim([-30 30]);title('Control Signal XY');xlabel('X');ylabel('Y');
% subplot(132);plot(YPred4);hold on;title('Control Signal X & Y');xlabel('Timesteps |X|&|Y|');ylabel('control');
% subplot(133);plot(YPred4(1:60),'m');hold on;title('Control Signal X');xlabel('Timesteps');ylabel('control');

%% error signal
Err = YPred5-YPred4;
figure(5);subplot(121);plot(Err,'Linewidth',2);hold on;title('Error Signal');xlabel('Timesteps |X|&|Y|');ylabel('control error');
%% retrain lstm2 with error

lr = [0.2 0.5 0.75 0.79 0.83 ...
    0.85 0.87 0.89 0.91 0.93 ...
    0.94 0.95 0.95 0.96 0.96...
    0.96 0.96 0.96 0.96 0.96 ];
lr1 = [0.2 0.3 0.4 0.5 0.6 ...
    0.7 0.8 0.9 0.91 0.93 ...
    0.94 0.95 0.95 0.96 0.96...
    0.96 0.96 0.96 0.96 0.96 ];

% Tra2 = [-lr(kk)*[YPred4(1:60)]+YPred5(1:60);YPred4(61:120)]; 
Tra2 = [lr1(kk)*Err(1:60);YPred4(61:120)];
figure(6);subplot(121);plot(Tra2,'Linewidth',2); hold on;title('Error Signal*learning Constant');xlabel('Timesteps |X|&|Y|');ylabel('control error');
subplot(122);plot(Tra2(1:60),'Linewidth',2);hold on;title('Error Signal*learning Constant X');xlabel('Timesteps |X|');ylabel('control error');
kk = kk+1;

%% online-within trial upfdation every m*10ms
m = 20; %timesteps
n = m-1;
for i = 1:m:60
    Tra3 = [Tra2(1:i+n);zeros(60-i-n,1);YPred4(61:120)];
   
    net    = lstm2ConToConCopy2;
    options = trainingOptions('adam', ...
    'MaxEpochs',30, ...
    'GradientThreshold',1, ...
    'InitialLearnRate',0.005, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropPeriod',125, ...
    'LearnRateDropFactor',0.2, ...
    'Verbose',0, ...
    'Plots','training-progress');
if isa(net,'SeriesNetwork')
    lgraph = layerGraph(net.Layers);
else
    lgraph = layerGraph(net);
end

net2 = trainNetwork(Tra3,Tra3,lgraph,options);
net  = net2;
lstm2ConToConCopy2 = net2;
save('lstm2ConToConCopy2.mat','lstm2ConToConCopy2');

%%
YPred5 = predict(net,in2);%in2
figure(7);
subplot(131);plot(YPred5(1:60),YPred5(61:120));hold on;xlim([-30 30]);title('AdaptNET-Control Signal XY');xlabel('X');ylabel('Y');
subplot(132);plot(YPred5(1:120));hold on;ylim([-30 30]);title('AdaptNET-Control Signal X & Y');xlabel('Timesteps |X|&|Y|');ylabel('control');
subplot(133);plot(YPred5(1:60),'Linewidth',2);hold on;ylim([-30 30]);title('AdaptNET-Control Signal X');xlabel('Timesteps');ylabel('control');
end
%% lstm3
net    = lstm3ConVel;
YPred6 = predict(net,YPred5);
figure(8);
subplot(131);plot(YPred6(1:60),YPred6(61:120),'k','Linewidth',2);hold on;title('Adapted Velocity Signal XY');xlabel('X');ylabel('Y');
subplot(132);plot(YPred6(1:120),'Linewidth',2);hold on;ylim([-0.3 0.3]);title('Adapted Velocity Signal X & Y');xlabel('Timesteps |X|&|Y|');ylabel('vel');
subplot(133);plot(YPred6(1:60),'Linewidth',2);hold on;ylim([-0.3 0.3]);title('Adapted Velocity Signal X');xlabel('Timesteps');ylabel('vel');

end
