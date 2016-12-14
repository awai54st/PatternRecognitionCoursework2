function chiIndex = chiSqrDist(trainingWineData, testingWineData)
    trainingSize = size(trainingWineData);
    testSize = size(testingWineData,1);
    chiIndex = zeros(testSize, 1);
    for i = 1:testSize
        eachTestData = testingWineData(i,:);    
        minValue = realmax;
        minIndex = 0;
        for j = 1:trainingSize
            eachTrainData = trainingWineData(j,:);
            value = 0;
            %For every bit in vector
            for k = 1:size(testingWineData,2)
                value = value + ((eachTestData(k) - eachTrainData(k))^2/(eachTestData(k)+eachTrainData(k)));
            end
            value = value / 2;
            if minValue > value
                minValue = value;
                minIndex = j;
            end    
        end
        chiIndex(i) = minIndex;
    end
end
