function res = LookupKMeansIndex(trainingIndex10, testIndex10, standardTrainingClassLabel, K)
    res = zeros(40,1);
    clusterLookup = zeros(K, 1);
    mergeIndex = [trainingIndex10,standardTrainingClassLabel'];
    
    for i = 1 : K
        tempLookup = zeros(size(mergeIndex, 1), 1);
        for j = 1 : size(mergeIndex, 1) 
            if mergeIndex(j,1) == i
                tempLookup(j) = mergeIndex(j,2);
            end
        end
        tempLookup = tempLookup(tempLookup~=0); %remove zeros from array
        clusterLookup(i) = mode(tempLookup);
    end
    
    for i = 1 : size(testIndex10, 1)
        res(i) = clusterLookup(testIndex10(i));
    end
end