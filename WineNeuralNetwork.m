clear all;
close all;
clc;

random_numbers = [65165, 6516861, 654161, 168168, 65188, 65461, 36245, 6121, 9635, 75515];

error_rate = zeros(20, 20);

training_function_index = 1;

%training_Fun = {'traingdm','traingdx','trainbfg','traincgb','traincgf','traincgp'};
training_Fun = {'trainbr'};
%transfer_Fun = {'logsig','tansig','purelin'};
transfer_Fun = {'softmax'};
for training_function = training_Fun
transfer_function_index = 1;
for transfer_function = transfer_Fun

for n_networks_1 = 19
%for n_networks_2 = 3:13
error = zeros(1, 10);
for iterations = 1:1

load wine.data.csv;
x = wine_data(:, 2:14)';
x = normr(x);
t = [(wine_data(:, 1)==1)'; (wine_data(:, 1)==2)'; (wine_data(:, 1)==3)'];

setdemorandstream(random_numbers(iterations))
net = patternnet(n_networks_1);
net.trainFcn = char(training_function);
%net.trainParam.mc = 0.6;
net.layers{1}.transferFcn = 'logsig';
net.layers{2}.transferFcn = 'softmax';
% net.trainParam.epochs = 300;
% net.trainParam.goal = 0;
% net.trainParam.lr = 0.1;
net.divideMode = 'sample';  % Divide up every sample
net.divideParam.trainRatio = 118/178;
%net.divideParam.valRatio = 20/178;
net.divideParam.valRatio = (1 - net.divideParam.trainRatio) / 3;
net.divideParam.testRatio = (1 - net.divideParam.trainRatio) * 2 / 3;
net.trainParam.showWindow = 1;
view(net)

[net,tr] = train(net,x,t);


testX = x(:,tr.testInd);
testT = t(:,tr.testInd);

testY = net(testX);

[c,cm] = confusion(testT,testY);

%fprintf('Percentage Correct Classification   : %f%%\n', 100*(1-c));
%fprintf('Percentage Incorrect Classification : %f%%\n', 100*c);

%plotroc(testT,testY)
error(iterations) = c;
end
%error_rate(n_networks_1, n_networks_2) = mean(error);
error_rate(training_function_index*3+transfer_function_index, n_networks_1) = mean(error);
%end
end
transfer_function_index = transfer_function_index + 1;
end

training_function_index = training_function_index + 1;
end

%https://eembdersler.files.wordpress.com/2010/09/2013911116-c3b6zgc3bcrc3a7elik-report.pdf
%http://neuroph.sourceforge.net/tutorials/wines1/WineClassificationUsingNeuralNetworks.html
%https://en.wikibooks.org/wiki/Artificial_Neural_Networks/Neural_Network_Basics
%http://stats.stackexchange.com/questions/181/how-to-choose-the-number-of-hidden-layers-and-nodes-in-a-feedforward-neural-netw

%%

clear all;
close all;
clc;

random_numbers = [65165, 6516861, 654161, 168168, 65188, 65461, 36245, 6121, 9635, 75515];

error_rate = zeros(20, 20);

training_function_index = 1;

%training_Fun = {'traingdm','traingdx','trainbfg','traincgb','traincgf','traincgp'};
training_Fun = {'traingdm','traingdx'};
%transfer_Fun = {'logsig','tansig','purelin'};
transfer_Fun = {'purelin'};
learning_RT = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0];
for training_function = training_Fun
transfer_function_index = 1;
%for transfer_function = transfer_Fun
%for learning_rate = learning_RT

for n_networks_1 = 1:9
learning_rate = learning_RT(n_networks_1);
%for n_networks_2 = 3:13
error = zeros(1, 10);
for iterations = 1:10

load wine.data.csv;
x = wine_data(:, 2:14)';
x = normr(x);
t = [(wine_data(:, 1)==1)'; (wine_data(:, 1)==2)'; (wine_data(:, 1)==3)'];

setdemorandstream(random_numbers(iterations))
net = patternnet(n_networks_1);
net.trainParam.lr = learning_rate;
net.trainFcn = char(training_function);
%net.trainParam.mc = 0.6;
net.layers{1}.transferFcn = 'purelin';
%net.layers{2}.transferFcn = 'logsig';
% net.trainParam.epochs = 300;
% net.trainParam.goal = 0;
% net.trainParam.lr = 0.1;
net.divideMode = 'sample';  % Divide up every sample
net.divideParam.trainRatio = 118/178;
net.divideParam.valRatio = 20/178;
net.divideParam.testRatio = 40/178;
net.trainParam.showWindow = 0;
%view(net)

[net,tr] = train(net,x,t);


testX = x(:,tr.testInd);
testT = t(:,tr.testInd);

testY = net(testX);

[c,cm] = confusion(testT,testY);

%fprintf('Percentage Correct Classification   : %f%%\n', 100*(1-c));
%fprintf('Percentage Incorrect Classification : %f%%\n', 100*c);

%plotroc(testT,testY)
error(iterations) = c;
end
%error_rate(n_networks_1, n_networks_2) = mean(error);
error_rate(training_function_index, n_networks_1) = mean(error);
%end
%end
transfer_function_index = transfer_function_index + 1;
end

training_function_index = training_function_index + 1;
end