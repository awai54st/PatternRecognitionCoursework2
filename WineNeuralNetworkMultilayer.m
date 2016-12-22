clear all;
close all;
clc;

random_numbers = [65165, 6516861, 654161, 168168, 65188, 65461, 36245, 6121, 9635, 75515];
error_rate = zeros(13, 13);

for n_networks_1 = 3:20
for n_networks_2 = 3:20
error = zeros(1, 10);
for iterations = 1:10

load wine.data.csv;
[x,t] = wine_dataset;

setdemorandstream(random_numbers(iterations))
net = patternnet([n_networks_1 n_networks_2]);
%net = patternnet(n_networks_1);
%net = cascadeforwardnet([n_networks 2]);
%net.trainParam.epochs = 300;
%net.trainParam.goal = 1e-5;
%net.trainParam.lr = 0.05;
net.trainFcn = 'trainbr';
net.layers{1}.transferFcn = 'logsig';
net.layers{2}.transferFcn = 'logsig';
net.layers{3}.transferFcn = 'softmax';
net.divideMode = 'sample';  % Divide up every sample
net.divideParam.trainRatio = 118/178;
net.divideParam.valRatio = 20/178;
net.divideParam.testRatio = 40/178;
net.trainParam.showWindow = 0;
%view(net)

[net,tr] = train(net,x,t);
%nntraintool
%plotperform(tr)

testX = x(:,tr.testInd);
testT = t(:,tr.testInd);

testY = net(testX);
%testIndices = vec2ind(testY)

%plotconfusion(testT,testY)

[c,cm] = confusion(testT,testY);

%fprintf('Percentage Correct Classification   : %f%%\n', 100*(1-c));
%fprintf('Percentage Incorrect Classification : %f%%\n', 100*c);

%plotroc(testT,testY)
error(iterations) = c;
end
error_rate(n_networks_1, n_networks_2) = mean(error);
%error_rate(n_networks_1, 1) = mean(error);
end
end