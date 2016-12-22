function res = LookupKMeansIndex(trainingIndex3, KMeansIndex10)
    res = zeros(40,1);
    for i = 1 : size(KMeansIndex10,1)
        res(i) = trainingIndex3(KMeansIndex10(i));
    end
end