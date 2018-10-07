clear
format compact
close all

load ../dataset/Salinas_hyperspectral %Load the Salinas hypercube called "Salinas_Image"
[p,n,l]=size(Salinas_Image); % p,n define the spatial resolution of the image, while l is the number of bands (number of features for each pixel)

load ../dataset/classification_labels 
% This file contains three arrays of dimension 22500x1 each, called
% "Training_Set", "Test_Set" and "Operational_Set". In order to bring them
% in an 150x150 image format we use the command "reshape" as follows:
Training_Set_Image=reshape(Training_Set, p,n); % In our case p=n=150 (spatial dimensions of the Salinas image).
Test_Set_Image=reshape(Test_Set, p,n);
Operational_Set_Image=reshape(Operational_Set, p,n);

% Constructing the 204xN array whose columns are the vectors corresponding to the
% N vectors (pixels) of the training set (similar codes can be used for
% the test and the operational sets).
Train=zeros(p,n,l); % This is a 3-dim array, which will contain nonzero values only for the training pixels
Test=zeros(p,n,l);
Operational=zeros(p,n,l);
for i=1:l
    %Multiply elementwise each band of the Salinas_Image with the mask 
    % "Training_Set_Image>0", which identifies only the training vectors.
    Train(:,:,i)=Salinas_Image(:,:,i).*(Training_Set_Image>0);
    Test(:,:,i)=Salinas_Image(:,:,i).*(Test_Set_Image>0);
    Operational(:,:,i)=Salinas_Image(:,:,i).*(Operational_Set_Image>0);
end

Train_array=[]; %This is the wanted 204xN array
Train_array_response=[]; % This vector keeps the label of each of the training pixels
Train_array_pos=[]; % This array keeps (in its rows) the position of the training pixels in the image.

Test_array=[]; %This is the wanted 204xN array
Test_array_response=[]; % This vector keeps the label of each of the training pixels
Test_array_pos=[]; % This array keeps (in its rows) the position of the training pixels in the image.

Operational_array=[]; %This is the wanted 204xN array
Operational_array_response=[]; % This vector keeps the label of each of the training pixels
Operational_array_pos=[]; % This array keeps (in its rows) the position of the training pixels in the image.
for i=1:p
    for j=1:n
        if(Training_Set_Image(i,j)>0) %Check if the (i,j) pixel is a training pixel
            Train_array=[Train_array squeeze(Train(i,j,:))];
            Train_array_response=[Train_array_response Training_Set_Image(i,j)];
            Train_array_pos=[Train_array_pos; i j];
        end
        if(Test_Set_Image(i,j)>0)
            Test_array=[Test_array squeeze(Test(i,j,:))];
            Test_array_response=[Test_array_response Test_Set_Image(i,j)];
            Test_array_pos=[Test_array_pos; i j];
        end
        if(Operational_Set_Image(i,j)>0)
            Operational_array=[Operational_array squeeze(Operational(i,j,:))];
            Operational_array_response=[Operational_array_response Operational_Set_Image(i,j)];
            Operational_array_pos=[Operational_array_pos; i j];
        end
    end
end

% Combine all sets into one (will be used to generate the images)
All_arrays = [Train_array Test_array Operational_array];
All_arrays_pos = [Train_array_pos' Test_array_pos' Operational_array_pos']';

m_hat=[];
S_hat=[];
% Estimate the maximum Likelihood parameters
for i=1:5
    for j=1:204
        X=Train_array(j,find(Train_array_response==i));
        [l,N]=size(X);
        m=(1/N)*sum(X')';
        S=zeros(l);
        for k=1:N
            S=S+(X(:,k)-m)*(X(:,k)-m)';
        end
        S=(1/N)*S;
        m_hat(i, j)=m;
        S_hat(i, j)=S;
    end
    m_hat(i)=m_hat(i)';S_hat(i)=S_hat(i)';
end

%%% Naive Bayes %%%
classified=naive_bayes_classifier(S_hat, m_hat, Test_array);

% Calculate the error rate, print the confusion matrix and use it to
% calculate the precision of the Naive Bayes classifier
true_labels=Test_array_response;
naive_error=sum(true_labels~=classified)/length(classified)
confusion_matrix = confusionmat(classified, true_labels)
precision = sum(trace(confusion_matrix))/sum(sum(confusion_matrix))

%%% Eucleidian Classifier %%%
m_hat=m_hat';

classified=euclidean_distance_classifier(m_hat, Test_array);

% Calculate the error rate, print the confusion matrix and use it to
% calculate the precision of the Eucleidian classifier
eucleidian_error=sum(true_labels~=classified)/length(classified)
confusion_matrix=confusionmat(classified, true_labels)
precision=sum(trace(confusion_matrix))/sum(sum(confusion_matrix))

%%% KNN %%%
[best_k, knn_test_response] = calculate_best_k(Train_array_response,Train_array_pos, Test_array_pos, Test_array_response);

classified= k_nn_classifier(Train_array_pos',Train_array_response',best_k,Test_array_pos');
knn_error = sum(classified~=Test_array_response)/length(knn_test_response)
confusion_matrix=confusionmat(classified, Test_array_response)
precision=sum(trace(confusion_matrix))/sum(sum(confusion_matrix))
best_k

%%% Generating Figures %%%

%%% Naive Bayes %%%
% Calculate the pdf for each element and for each class, sum the results
% and pick the index of the max value
m_hat=m_hat';
perFeature=[];
naive_probs=[];
for i=1:5
    for j=1:204
        perFeature(i,j,:)=normpdf(All_arrays(j,:),m_hat(i,j),sqrt(S_hat(i,j)));
    end
end
naive_probs=sum(perFeature,2);
naive_probs = squeeze(naive_probs);
[max_value, idx]=max(naive_probs);
classified=idx;

% Create figure with the classified elements
Result_Fig = zeros(p,n);
for i=1:length(All_arrays_pos)
    Result_Fig(All_arrays_pos(i,1),All_arrays_pos(i,2)) = classified(1,i);
end
figure('Name', 'Naive Bayes'); imagesc(Result_Fig);

%%% Eucleidian %%%
m_hat=m_hat';
[l,c]=size(m_hat);
[l,N]=size(All_arrays);

% Calculate the squared Eucleidian distance
for i=1:N
    for j=1:c
        distance(j)=sqrt((All_arrays(:,i)-m_hat(:,j))'*(All_arrays(:,i)-m_hat(:,j)));
    end
    [num,idx(i)]=min(distance);
end
classified=idx;

% Create figure with the classified elements
Result_Fig = zeros(p,n);
for i=1:length(All_arrays_pos)
    Result_Fig(All_arrays_pos(i,1),All_arrays_pos(i,2)) = classified(1,i);
end
figure('Name','Eucleidian Distance'); imagesc(Result_Fig);

%%% KNN %%%
classified= k_nn_classifier(Train_array_pos',Train_array_response',best_k,All_arrays_pos');

% Create figure with the classified elements
Result_Fig = zeros(p,n);
for i=1:length(All_arrays_pos)
    Result_Fig(All_arrays_pos(i,1),All_arrays_pos(i,2)) = classified(1,i);
end
figure('Name','KNN'); imagesc(Result_Fig);

% Create figure with the true values of all sets
Result_Fig = zeros(p,n);
for i=1:length(Test_array_pos)
    Result_Fig(Test_array_pos(i,1),Test_array_pos(i,2)) = Test_array_response(1,i);
end
for i=1:length(Train_array_pos)
    Result_Fig(Train_array_pos(i,1),Train_array_pos(i,2)) = Train_array_response(1,i);
end
for i=1:length(Operational_array_pos)
    Result_Fig(Operational_array_pos(i,1),Operational_array_pos(i,2)) = Operational_array_response(1,i);
end
figure('Name','Combined Set'); imagesc(Result_Fig);
