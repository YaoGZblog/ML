function [y_predict,performance,Imp] = ygz_predict(x,y,kfolds,Pre_Method,Method)
%% split data
nsubs = size(x,1);
randinds = randperm(nsubs);
y_predict = zeros(nsubs,1);

for leftout = kfolds:-1:1
    if kfolds == nsubs
        testinds = randinds(leftout);
        traininds = setdiff(randinds,testinds);
    elseif kfolds ==1
    else
        testinds=randinds(leftout:kfolds:nsubs);
        traininds=setdiff(randinds,testinds);
    end
    x_train = x(traininds,:);
    y_train = y(traininds,:);
    x_test = x(testinds,:);
    %%  pre_process data
    if strcmp(Pre_Method,'Normalize')
        % Normalize
        MeanValue = mean(x_train);
        StandardDeviation = std(x_train);
        [~,nvar] =size(x_train);
        for j = 1:nvar
            x_train(:,j) = (x_train(:, j) - MeanValue(j)) / StandardDeviation(j);
        end
        MeanValue_New = repmat(MeanValue, length(testinds), 1);
        StandardDeviation_New = repmat(StandardDeviation, length(testinds), 1);
        x_test = (x_test - MeanValue_New) ./ StandardDeviation_New;
    elseif strcmp(Pre_Method, 'Scale')
        % Scaling to [0 1]
        MinValue = min(x_train);
        MaxValue = max(x_train);
        [~, nvar] = size(x_train);
        for j = 1:nvar
            x_train(:, j) = (x_train(:, j) - MinValue(j)) / (MaxValue(j) - MinValue(j));
        end
        MaxValue_New = repmat(MaxValue, length(testinds), 1);
        MinValue_New = repmat(MinValue, length(testinds), 1);
        x_test = (x_test - MinValue_New) ./ (MaxValue_New - MinValue_New);
    end

    %% Train Model
    [mdl,importance] = ygz_train(x_train,y_train);
    %% Test Model
    [y_predict(testinds,1)] = ygz_test(x_test,mdl);
    %% Assess Performance
    if kfolds == 1
    yt = y(testinds);
    performance(:,1) = corr(y_predict(testinds,:),yt);
    performance(:,2) = sqrt(mean((y_predict(testinds,:)- [yt]).^2)); % root-mean-squared-error (RMSE)
    else
    performance(:,1) = corr(y_predict,y);
    performance(:,2) = sqrt(mean((y_predict-[y]).^2)); % root-mean-squared-error (RMSE)

    end
performance(isnan(performance(:,1)),1) = -1;
performance(isnan(performance(:,2)),2) = 1;
Imp(leftout,:) = importance;
end
function [mdl,importance] = ygz_train(x,y)
    if strcmp(Method,'tree')
        mdl = fitrtree(x,y);
        importance = predictorImportance(mdl);
    elseif strcmp(Method,'liner')
        mdl = fitlm(x,y);
        importance = mdl.Coefficients.Estimate(2:end);
    end
end
function [y_predict]=ygz_test(x,mdl)
    y_predict = predict(mdl,x);
end
end
