function plotNNErrorRate
sqErrorMatrix = [0.078,0.06525,0.10475,0.112,0.03475];
cityErrorMatrix = [0.078,0.06825,0.104,0.1125,0.07075];
cosErrorMatrix = [0.1165,0.00875,0.113,0.12075,0];
corrErrorMatrix = [0.10475,0.0105,0.104,0.10425,0.01375];
%xLabel = ['Sqeuclidean','Cityblock','Cosine','Correlation','Mahalanobis'];
xLabel{1} = 'Sqeuclidean';
xLabel{2} = 'Cityblock';
xLabel{3} = 'Cosine';
xLabel{4} = 'Correlation';
xLabel{5} = 'Mahalanobis';

figure;
bar(sqErrorMatrix')
set(gca,'xticklabel', xLabel); 
ylabel('Classification Error');

figure;
bar(cityErrorMatrix')
set(gca,'xticklabel', xLabel); 
ylabel('Classification Error');

figure;
bar(cosErrorMatrix')
set(gca,'xticklabel', xLabel); 
ylabel('Classification Error');

figure;
bar(corrErrorMatrix')
set(gca,'xticklabel', xLabel); 
ylabel('Classification Error');
end
