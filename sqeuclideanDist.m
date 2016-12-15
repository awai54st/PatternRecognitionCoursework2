function sqeuclideanIndex = sqeuclideanDist(x, testingWineData)
    xSize = size(x);
    testSize = size(testingWineData,1);
    sqeuclideanIndex = zeros(testSize, 1);
    for i = 1:testSize
        eachTestData = testingWineData(i,:);    
        minValue = realmax;
        minIndex = 0;
        for j = 1:xSize
            eachClusterData = x(j,:);
            value = (eachTestData - eachClusterData) * (eachTestData - eachClusterData)';
            if minValue > value
                minValue = value;
                minIndex = j;
            end    
        end
        sqeuclideanIndex(i) = minIndex;
    end
end
