function [classified]=naive_bayes_classifier(S_hat, m_hat, array)

% Calculate the pdf for each element and for each class, sum the results
% and pick the index of the max value (where the index refers to the class
% to pick)
perFeature=[];
naive_probs=[];
for i=1:5
    for j=1:204
        perFeature(i,j,:)=normpdf(array(j,:),m_hat(i,j),sqrt(S_hat(i,j)));
    end
end
naive_probs=sum(perFeature,2);
naive_probs = squeeze(naive_probs);
[max_value, idx]=max(naive_probs);
classified=idx;