%%Q1)A) la lalalala lalala
load wine.data.csv;
classIdentifier = wine_data(:,1);
wineData = wine_data(:,2:14);
normWineData = normr(wineData);

wineDataC1 = wineData(1:59,:);
wineDataC2 = wineData(60:130,:);
wineDataC3 = wineData(131:178,:);
normWineDataC1 = normr(wineDataC1);
normWineDataC2 = normr(wineDataC2);
normWineDataC3 = normr(wineDataC3);

%%Q1)B) wakakakakakaka 

covMatrixAll = cov(wineData);
covMatrixAllNorm = cov(normWineData);

covMatrixC1 = cov(wineDataC1);
covMatrixC2 = cov(wineDataC2);
covMatrixC3 = cov(wineDataC3);
covMatrixC1Norm = cov(normWineDataC1);
covMatrixC2Norm = cov(normWineDataC2);
covMatrixC3Norm = cov(normWineDataC3);

%%Q1)C) 
