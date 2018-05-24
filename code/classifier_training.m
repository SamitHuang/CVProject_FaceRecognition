%% Train a linear classifier from the positive and negative examples with a call to vl_trainsvm

function [w, b] = classifier_training(features_pos, features_neg, feature_params)
    features = [features_pos',features_neg']; % one column per sample
    labels = [1* ones(size(features_pos,1),1); -1 * ones(size(features_neg,1),1)];  % binary (-1 or +1) label 
    
    [w, b] = vl_svmtrain(features, labels, 1e-4); %regularization coefficient LAMBDA = 1e-4
    
    