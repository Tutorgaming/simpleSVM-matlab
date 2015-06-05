%LIBSVM
%============
% Inputs
%============
%Read Input File Using LIBSVM-READ
%Input Data must be SCALED to the range [-1,1]
    [class, data_instance] = libsvmread('iris.scale');

    [N D] = size(data_instance);

% Determine the train and test index
    trainIndex = zeros(N,1); trainIndex(1:N) = 1;
    trainData = data_instance(trainIndex==1,:);
    trainLabel = class(trainIndex==1,:);
    testIndex = zeros(N,1); testIndex(1:N) = 1;
    testData = data_instance(testIndex==1,:);
    testLabel = class(testIndex==1,:);

%RAW DATA PLOT
    figure(1);
    scatter(trainData(:,1),trainData(:,2),[],trainLabel(:,1),'+','linewidth',2);
    hold on;

%============
% Processing
%============

% Train the SVM
    model = svmtrain(trainLabel, trainData, '-s 0 -t 0 -c 100');
% Use the SVM model to classify the data
    [predict_label, accuracy, prob_values] = svmpredict(testLabel, testData, model, '-b 1'); % run the SVM model on the test data

    w = model.SVs' * model.sv_coef;
    b = -model.rho;
    if (model.Label(1) == -1)
        w = -w; b = -b;
    end
    y_hat = sign(w'*trainData' + b);

%============
% Outputs
%============

    sv = full(model.SVs);
% plot support vectors
    plot(sv(:,1),sv(:,2),'ko', 'MarkerSize', 10);

% plot decision boundary
    plot_x = linspace(min(trainData(:,1)), max(trainData(:,1)), 30);
    plot_y = (-1/w(2))*(w(1)*plot_x + b);
    plot(plot_x, plot_y, 'k-', 'LineWidth', 1)