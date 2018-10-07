function [best_k, knn_test_response]=calculate_best_k(Train_array_response,Train_array_pos, Test_array_pos, Test_array_response)

% Split the training data into 5 parts using cross-validation
indices = crossvalind('Kfold', Train_array_response, 5);
best_k=-1;
lowest_error=-1;
% Pick the best value for k by training the KNN classifier 5 times each
% time different parts of the split dataset. Calculate each time the error
% rate and pick the k that gives the lowest error rate
for k=1:2:17
    avg_err=0;
    for i=1:5
        knn_train=Train_array_pos(find(indices ~= i),:);
        knn_train_response=Train_array_response(:,find(indices ~= i));
        knn_test=Test_array_pos(find(indices == i),:);
        knn_test_response=Test_array_response(:,find(indices == i));

        knn_train=knn_train';
        knn_test=knn_test';

        result = Knn(k,knn_train,knn_train_response,knn_test);
        pr_err = sum(result~=knn_test_response)/length(knn_test_response');
        avg_err=avg_err+pr_err;
    end
    avg_err=avg_err/5; % use the average error rate for the current value of k
    if((best_k==-1) || (avg_err<lowest_err))
        best_k=k;
        lowest_err=avg_err;
    end
end
[best_k, knn_test_response];