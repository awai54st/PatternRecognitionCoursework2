%This file is specialized for Q2 when K = 3
clear all;
close all;
clc;
%% Q1)A) la lalalala lalala
load wine.data.csv;
errorMatrix = ones(5, 100);
for i = 1:100
classIdentifier = wine_data(:,1);
wineData = wine_data(:,2:14);
normWineData = normc(wineData);

wineDataC1 = wineData(1:59,:);
wineDataC2 = wineData(60:130,:);
wineDataC3 = wineData(131:178,:);
normWineDataC1 = normWineData(1:59,:);
normWineDataC2 = normWineData(60:130,:);
normWineDataC3 = normWineData(131:178,:);

randWineDataC1 = wineDataC1(randperm(59),:);
randWineDataC2 = wineDataC2(randperm(71),:);
randWineDataC3 = wineDataC3(randperm(48),:);

randNormWineDataC1 = normWineDataC1(randperm(59),:);
randNormWineDataC2 = normWineDataC2(randperm(71),:);
randNormWineDataC3 = normWineDataC3(randperm(48),:);

%Split data into training, validation, testing set
trainingWineData = [randWineDataC1(1:39,:); randWineDataC2(1:47,:); randWineDataC3(1:32,:)];
validationWineData = [randWineDataC1(40:46,:); randWineDataC2(48:55,:); randWineDataC3(33:37,:)];
testingWineData = [randWineDataC1(47:59,:); randWineDataC2(56:71,:); randWineDataC3(38:48,:)];

%Split data into unit norm L2 training, validation, testing set
normTrainingWineData = [randNormWineDataC1(1:39,:); randNormWineDataC2(1:47,:); randNormWineDataC3(1:32,:)];
normValidationWineData = [randNormWineDataC1(40:46,:); randNormWineDataC2(48:55,:); randNormWineDataC3(33:37,:)];
normTestingWineData = [randNormWineDataC1(47:59,:); randNormWineDataC2(56:71,:); randNormWineDataC3(38:48,:)];

%Class labels
trainingDataClassLabel = [ones(1,39), 2*ones(1,47), 3*ones(1,32)];
validationDataClassLabel = [ones(1,7), 2*ones(1,8), 3*ones(1,5)];
testingDataClassLabel = [ones(1,13), 2*ones(1,16), 3*ones(1,11)];

trainingWineDataC1 = randWineDataC1(1:39,:);
trainingWineDataC2 = randWineDataC2(1:47,:);
trainingWineDataC3 = randWineDataC3(1:32,:);
trainingNormWineDataC1 = randNormWineDataC1(1:39,:);
trainingNormWineDataC2 = randNormWineDataC2(1:47,:);
trainingNormWineDataC3 = randNormWineDataC3(1:32,:);

%% Q1)B) wakakakakakaka 

covMatrixAll = cov(trainingWineData);
covMatrixAllNorm = cov(normTrainingWineData);

covMatrixC1 = cov(trainingWineDataC1);
covMatrixC2 = cov(trainingWineDataC2);
covMatrixC3 = cov(trainingWineDataC3);
covMatrixC1Norm = cov(trainingNormWineDataC1);
covMatrixC2Norm = cov(trainingNormWineDataC2);
covMatrixC3Norm = cov(trainingNormWineDataC3);

%% Q2) Kmeans clustering
    %Sqeuclidian
    tic;
    [sqeuclidianTrainingIndex, sqeuclidianTrainingClassCentre] = kmeans(normTrainingWineData, 3, 'Replicate', 100);
    sqTime = toc;
    silhouette(normTrainingWineData,sqeuclidianTrainingIndex);
    sqeuclidianKMeansIndex = sqeuclideanDist(sqeuclidianTrainingClassCentre, normTestingWineData);
    sqeuclidianErr = classificationErr(sqeuclidianKMeansIndex, 3);
    errorMatrix(1, i) = sqeuclidianErr;
    % figure;
    % scatter(normTrainingWineData(:,7),normTrainingWineData(:,11));
    % hold on;
    % scatter(sqeuclidianTrainingClassCentre(2,7),sqeuclidianTrainingClassCentre(2,11));
    % hold off;

    % %cityblock
    tic;
    [cityblockTrainingIndex, cityblockTrainingClassCentre] = kmeans(normTrainingWineData, 3, 'Replicate', 100, 'Distance', 'cityblock'); 
    cityTime = toc;
    cityblockKMeansIndex = knnsearch(cityblockTrainingClassCentre, normTestingWineData, 'Distance', 'cityblock');
    cityblockErr = classificationErr(cityblockKMeansIndex, 3);
    errorMatrix(2, i) = cityblockErr;

    %Cosine
    tic;
    [cosineTrainingIndex, cosineTrainingClassCentre] = kmeans(normTrainingWineData, 3, 'Replicate', 100, 'Distance', 'cosine'); 
    cosTime = toc;
    cosineKMeansIndex = knnsearch(cosineTrainingClassCentre, normTestingWineData, 'Distance', 'cosine');
    cosineErr = classificationErr(cosineKMeansIndex, 3);
    errorMatrix(3, i) = cosineErr;

    %Correlation
    tic;
    [correlationTrainingIndex, correlationTrainingClassCentre] = kmeans(normTrainingWineData, 3, 'Replicate', 100, 'Distance', 'correlation'); 
    corrTime = toc;
    correlationKMeansIndex = knnsearch(correlationTrainingClassCentre, normTestingWineData, 'Distance', 'correlation');
    correlationErr = classificationErr(correlationKMeansIndex, 3);
    errorMatrix(4, i) = correlationErr; 
    
    
    %Mahalanobias
    upperTriag = chol(covMatrixAllNorm);
    transformedTrainingWineData = normTrainingWineData * upperTriag;
    [mahalanobisTrainingIndex, mahalanobisTrainingClassCentre] = kmeans(transformedTrainingWineData, 3, 'Replicates', 100); 
    mahalanobisKMeansIndex = knnsearch(mahalanobisTrainingClassCentre, normTestingWineData * upperTriag, 'Distance', 'mahalanobis', 'Cov', covMatrixAllNorm);
    mahalanobisErr = classificationErr(mahalanobisKMeansIndex, 3);
    errorMatrix(5, i) = mahalanobisErr;
end
meanErrorMatrix = mean(errorMatrix,2);


%Draw training data clustering 
DrawKmeansClustering(normTrainingWineData, randNormWineDataC1, randNormWineDataC2, randNormWineDataC3, sqeuclidianTrainingIndex, sqeuclidianTrainingClassCentre, cityblockTrainingIndex, cityblockTrainingClassCentre, cosineTrainingIndex, cosineTrainingClassCentre, correlationTrainingIndex, correlationTrainingClassCentre, mahalanobisTrainingIndex, mahalanobisTrainingClassCentre);
