function DrawKmeansClustering(normTrainingWineData, randNormWineDataC1, randNormWineDataC2, randNormWineDataC3, sqeuclidianTrainingIndex, sqTrainingClassCentre, cityblockTrainingIndex, cityTrainingClassCentre, cosineTrainingIndex, cosTrainingClassCentre, correlationTrainingIndex, corrTrainingClassCentre,  mahalanobisTrainingIndex, mahalanobisTrainingClassCentre)
    %Draw Kmeans clustered training data
    KmeansTrainingDataC1 = zeros(1,13);
    KmeansTrainingDataC2 = zeros(1,13);
    KmeansTrainingDataC3 = zeros(1,13);
    for j = 1:size(sqeuclidianTrainingIndex,1)
        if sqeuclidianTrainingIndex(j) == 1
            if all(KmeansTrainingDataC1) == 0
                KmeansTrainingDataC1 = normTrainingWineData(j,:);
            else
                KmeansTrainingDataC1 = [KmeansTrainingDataC1; normTrainingWineData(j,:)];
            end
        elseif sqeuclidianTrainingIndex(j) == 2
            if all(KmeansTrainingDataC2) == 0
                KmeansTrainingDataC2 = normTrainingWineData(j,:);
            else
                KmeansTrainingDataC2 = [KmeansTrainingDataC2; normTrainingWineData(j,:)];
            end
        elseif sqeuclidianTrainingIndex(j) == 3
            if all(KmeansTrainingDataC3) == 0
                KmeansTrainingDataC3 = normTrainingWineData(j,:);
            else
                KmeansTrainingDataC3 = [KmeansTrainingDataC3; normTrainingWineData(j,:)];
            end
        end
    end

    figure;
    scatter (KmeansTrainingDataC1(:, 7), KmeansTrainingDataC1(:, 11));
    hold on;
    scatter (KmeansTrainingDataC2(:, 7), KmeansTrainingDataC2(:, 11));
    hold on;
    scatter (KmeansTrainingDataC3(:, 7), KmeansTrainingDataC3(:, 11));
    hold on;
    scatter (sqTrainingClassCentre(1,7), sqTrainingClassCentre(1,11));
    hold on;
    scatter (sqTrainingClassCentre(2,7), sqTrainingClassCentre(2,11));
    hold on;
    scatter (sqTrainingClassCentre(3,7), sqTrainingClassCentre(3,11));
    hold off;
    legend('Class 1','Class 2','Class 3', 'Class1Centre', 'Class2Centre', 'Class3Centre') ;
    xlabel('Dimension 7');
    ylabel('Dimension 11');
    title('Kmeans Training Data Clustering on sqeuclidean');

    %Draw Kmeans clustered training data cityblock
    KmeansTrainingDataC1 = zeros(1,13);
    KmeansTrainingDataC2 = zeros(1,13);
    KmeansTrainingDataC3 = zeros(1,13);
    for j = 1:size(cityblockTrainingIndex,1)
        if cityblockTrainingIndex(j) == 1
            if all(KmeansTrainingDataC1) == 0
                KmeansTrainingDataC1 = normTrainingWineData(j,:);
            else
                KmeansTrainingDataC1 = [KmeansTrainingDataC1; normTrainingWineData(j,:)];
            end
        elseif cityblockTrainingIndex(j) == 2
            if all(KmeansTrainingDataC2) == 0
                KmeansTrainingDataC2 = normTrainingWineData(j,:);
            else
                KmeansTrainingDataC2 = [KmeansTrainingDataC2; normTrainingWineData(j,:)];
            end
        elseif cityblockTrainingIndex(j) == 3
            if all(KmeansTrainingDataC3) == 0
                KmeansTrainingDataC3 = normTrainingWineData(j,:);
            else
                KmeansTrainingDataC3 = [KmeansTrainingDataC3; normTrainingWineData(j,:)];
            end
        end
    end

    figure;
    scatter (KmeansTrainingDataC1(:, 7), KmeansTrainingDataC1(:, 11));
    hold on;
    scatter (KmeansTrainingDataC2(:, 7), KmeansTrainingDataC2(:, 11));
    hold on;
    scatter (KmeansTrainingDataC3(:, 7), KmeansTrainingDataC3(:, 11));
    hold on;
    scatter (cityTrainingClassCentre(1,7), cityTrainingClassCentre(1,11));
    hold on;
    scatter (cityTrainingClassCentre(2,7), cityTrainingClassCentre(2,11));
    hold on;
    scatter (cityTrainingClassCentre(3,7), cityTrainingClassCentre(3,11));
    hold off;
    legend('Class 1','Class 2','Class 3', 'Class1Centre', 'Class2Centre', 'Class3Centre') ;
    xlabel('Dimension 7');
    ylabel('Dimension 11');
    title('Kmeans Training Data Clustering on cityblock');

    %Draw Kmeans clustered training data cosineTrainingIndex
    KmeansTrainingDataC1 = zeros(1,13);
    KmeansTrainingDataC2 = zeros(1,13);
    KmeansTrainingDataC3 = zeros(1,13);
    for j = 1:size(cosineTrainingIndex,1)
        if cosineTrainingIndex(j) == 1
            if all(KmeansTrainingDataC1) == 0
                KmeansTrainingDataC1 = normTrainingWineData(j,:);
            else
                KmeansTrainingDataC1 = [KmeansTrainingDataC1; normTrainingWineData(j,:)];
            end
        elseif cosineTrainingIndex(j) == 2
            if all(KmeansTrainingDataC2) == 0
                KmeansTrainingDataC2 = normTrainingWineData(j,:);
            else
                KmeansTrainingDataC2 = [KmeansTrainingDataC2; normTrainingWineData(j,:)];
            end
        elseif cosineTrainingIndex(j) == 3
            if all(KmeansTrainingDataC3) == 0
                KmeansTrainingDataC3 = normTrainingWineData(j,:);
            else
                KmeansTrainingDataC3 = [KmeansTrainingDataC3; normTrainingWineData(j,:)];
            end
        end
    end

    figure;
    scatter (KmeansTrainingDataC1(:, 7), KmeansTrainingDataC1(:, 11));
    hold on;
    scatter (KmeansTrainingDataC2(:, 7), KmeansTrainingDataC2(:, 11));
    hold on;
    scatter (KmeansTrainingDataC3(:, 7), KmeansTrainingDataC3(:, 11));
    hold on;
    scatter (cosTrainingClassCentre(1,7), cosTrainingClassCentre(1,11));
    hold on;
    scatter (cosTrainingClassCentre(2,7), cosTrainingClassCentre(2,11));
    hold on;
    scatter (cosTrainingClassCentre(3,7), cosTrainingClassCentre(3,11));
    hold off;
    legend('Class 1','Class 2','Class 3', 'Class1Centre', 'Class2Centre', 'Class3Centre') ;
    xlabel('Dimension 7');
    ylabel('Dimension 11');
    title('Kmeans Training Data Clustering on cosine');

    %Draw Kmeans clustered training data correlationTrainingIndex
    KmeansTrainingDataC1 = zeros(1,13);
    KmeansTrainingDataC2 = zeros(1,13);
    KmeansTrainingDataC3 = zeros(1,13);
    for j = 1:size(correlationTrainingIndex,1)
        if correlationTrainingIndex(j) == 1
            if all(KmeansTrainingDataC1) == 0
                KmeansTrainingDataC1 = normTrainingWineData(j,:);
            else
                KmeansTrainingDataC1 = [KmeansTrainingDataC1; normTrainingWineData(j,:)];
            end
        elseif correlationTrainingIndex(j) == 2
            if all(KmeansTrainingDataC2) == 0
                KmeansTrainingDataC2 = normTrainingWineData(j,:);
            else
                KmeansTrainingDataC2 = [KmeansTrainingDataC2; normTrainingWineData(j,:)];
            end
        elseif correlationTrainingIndex(j) == 3
            if all(KmeansTrainingDataC3) == 0
                KmeansTrainingDataC3 = normTrainingWineData(j,:);
            else
                KmeansTrainingDataC3 = [KmeansTrainingDataC3; normTrainingWineData(j,:)];
            end
        end
    end

    figure;
    scatter (KmeansTrainingDataC1(:, 7), KmeansTrainingDataC1(:, 11));
    hold on;
    scatter (KmeansTrainingDataC2(:, 7), KmeansTrainingDataC2(:, 11));
    hold on;
    scatter (KmeansTrainingDataC3(:, 7), KmeansTrainingDataC3(:, 11));
    hold on;
    scatter (corrTrainingClassCentre(1,7), corrTrainingClassCentre(1,11));
    hold on;
    scatter (corrTrainingClassCentre(2,7), corrTrainingClassCentre(2,11));
    hold on;
    scatter (corrTrainingClassCentre(3,7), corrTrainingClassCentre(3,11));
    hold off;
    legend('Class 1','Class 2','Class 3', 'Class1Centre', 'Class2Centre', 'Class3Centre') ;
    xlabel('Dimension 7');
    ylabel('Dimension 11');
    title('Kmeans Training Data Clustering on correlation');

    %Draw actual training data
    figure;
    scatter (randNormWineDataC1(:, 7), randNormWineDataC1(:, 11));
    hold on;
    scatter (randNormWineDataC2(:, 7), randNormWineDataC2(:, 11));
    hold on;
    scatter (randNormWineDataC3(:, 7), randNormWineDataC3(:, 11));
    hold off;
    legend('Class 1','Class 2','Class 3') ;
    xlabel('Dimension 7');
    ylabel('Dimension 11');
    title('Actual Training Data Clustering');
    
    %Draw Kmeans clustered training data mahalanobisTrainingIndex
    KmeansTrainingDataC1 = zeros(1,13);
    KmeansTrainingDataC2 = zeros(1,13);
    KmeansTrainingDataC3 = zeros(1,13);
    for j = 1:size(mahalanobisTrainingIndex,1)
        if mahalanobisTrainingIndex(j) == 1
            if all(KmeansTrainingDataC1) == 0
                KmeansTrainingDataC1 = normTrainingWineData(j,:);
            else
                KmeansTrainingDataC1 = [KmeansTrainingDataC1; normTrainingWineData(j,:)];
            end
        elseif mahalanobisTrainingIndex(j) == 2
            if all(KmeansTrainingDataC2) == 0
                KmeansTrainingDataC2 = normTrainingWineData(j,:);
            else
                KmeansTrainingDataC2 = [KmeansTrainingDataC2; normTrainingWineData(j,:)];
            end
        elseif mahalanobisTrainingIndex(j) == 3
            if all(KmeansTrainingDataC3) == 0
                KmeansTrainingDataC3 = normTrainingWineData(j,:);
            else
                KmeansTrainingDataC3 = [KmeansTrainingDataC3; normTrainingWineData(j,:)];
            end
        end
    end
    
    figure;
    scatter (KmeansTrainingDataC1(:, 7), KmeansTrainingDataC1(:, 11));
    hold on;
    scatter (KmeansTrainingDataC2(:, 7), KmeansTrainingDataC2(:, 11));
    hold on;
    scatter (KmeansTrainingDataC3(:, 7), KmeansTrainingDataC3(:, 11));
    hold on;
    scatter (mahalanobisTrainingClassCentre(1,7), mahalanobisTrainingClassCentre(1,11));
    hold on;
    scatter (mahalanobisTrainingClassCentre(2,7), mahalanobisTrainingClassCentre(2,11));
    hold on;
    scatter (mahalanobisTrainingClassCentre(3,7), mahalanobisTrainingClassCentre(3,11));
    hold off;
    legend('Class 1','Class 2','Class 3', 'Class1Centre', 'Class2Centre', 'Class3Centre') ;
    xlabel('Dimension 7');
    ylabel('Dimension 11');
    title('Kmeans Training Data Clustering on mahalanobis');
end