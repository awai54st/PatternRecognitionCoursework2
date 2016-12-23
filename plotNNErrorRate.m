function plotNNErrorRate
close all;
sqErrorMatrix = [7.8,6.525,10.475,11.2,3.475];
cityErrorMatrix = [7.8,6.825,10.4,11.25,7.075];
cosErrorMatrix = [11.65,0.875,11.3,12.075,0];
corrErrorMatrix = [10.475,1.05,10.4,10.425,1.375];
%xLabel = ['Sqeuclidean','Cityblock','Cosine','Correlation','Mahalanobis'];
xLabel{1} = 'Sqeuclidean';
xLabel{2} = 'Cityblock';
xLabel{3} = 'Cosine';
xLabel{4} = 'Correlation';
xLabel{5} = 'Mahalanobis';

figure;
bar(sqErrorMatrix')
title('KNN Classification based on Sqeuclidean Metric for Kmeans');
set(gca,'xticklabel', xLabel); 
ylabel('Classification Error/per cent');

figure;
bar(cityErrorMatrix')
title('KNN Classification based on Cityblock Metric for Kmeans');
set(gca,'xticklabel', xLabel); 
ylabel('Classification Error/per cent');

figure;
bar(cosErrorMatrix')
title('KNN Classification based on Cosine Metric for Kmeans');
set(gca,'xticklabel', xLabel); 
ylabel('Classification Error/per cent');

figure;
bar(corrErrorMatrix')
title('KNN Classification based on Correlation Metric for Kmeans');
set(gca,'xticklabel', xLabel); 
ylabel('Classification Error/per cent');

errMatrix = [9.6000, 6.9250, 9.4750, 8.8250, 9.9750];
figure;
bar(errMatrix')
title('KNN Classification based on corresponding metrics for Kmeans,K=10');
set(gca,'xticklabel', xLabel); 
ylabel('Classification Error/per cent');



end
