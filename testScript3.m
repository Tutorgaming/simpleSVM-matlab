%LIBSVM
%============
% Inputs
%============
[heart_scale_label, heart_scale_inst] = libsvmread('heart_scale');

[N D] = size(heart_scale_inst);

% Determine the train and test index
trainIndex = zeros(N,1); trainIndex(1:200) = 1;
testIndex = zeros(N,1); testIndex(201:N) = 1;
trainData = heart_scale_inst(trainIndex==1,:);
trainLabel = heart_scale_label(trainIndex==1,:);
testData = heart_scale_inst(testIndex==1,:);
testLabel = heart_scale_label(testIndex==1,:);

% Train the SVM
model = svmtrain(trainLabel, trainData, '-c 1 -g 0.07 -b 1');
% Use the SVM model to classify the data
[predict_label, accuracy, prob_values] = svmpredict(testLabel, testData, model, '-b 1'); % run the SVM model on the test data



% ================================
% ===== Showing the results ======
% ================================

% Assign color for each class
% colorList = generateColorList(2); % This is my own way to assign the color...don't worry about it
colorList = prism(100);

% true (ground truth) class
trueClassIndex = zeros(N,1);
trueClassIndex(heart_scale_label==1) = 1; 
trueClassIndex(heart_scale_label==-1) = 2;
colorTrueClass = colorList(trueClassIndex,:);

% Reduce the dimension from 13D to 2D
distanceMatrix = pdist(heart_scale_inst,'euclidean');
newCoor = mdscale(distanceMatrix,2);

% Plot the whole data set
x = newCoor(:,1);
y = newCoor(:,2);
patchSize = 30; %max(prob_values,[],2);
colorTrueClassPlot = colorTrueClass;
figure; scatter(x,y,patchSize,colorTrueClassPlot,'filled');
title('whole data set');


% figure(1);
% plot(data_instance);
% 
% %============
% % Processing
% %============
% model = svmtrain(class, data_instance, ...
%     '-c 1 -g 0.07 -b 1');
