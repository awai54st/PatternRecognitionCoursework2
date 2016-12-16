function err = classificationErr(testingIndex, k)
    testingClassLabel = zeros(1,k);
    classSize = [13, 16, 11];
    index = 1;
    for j = 1:k
        testingClassLabel(j) = mode(testingIndex(index:classSize(j)+index-1,:));
        index = classSize(j)+index;
    end
    standardLabel = ([testingClassLabel(1)*ones(1,13), testingClassLabel(2)*ones(1,16), testingClassLabel(3)*ones(1,11)])';
    
    count = 0;
    for j = 1:size(testingIndex,1)
        if testingIndex(j) == standardLabel(j)
            count = count + 1;
        end
    end
    
    err = 1 - count / size(testingIndex,1);
end
