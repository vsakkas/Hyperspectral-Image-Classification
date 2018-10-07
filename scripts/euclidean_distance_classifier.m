function [classified]=euclidean_distance_classifier(m_hat, Test_array)

[l,c]=size(m_hat);
[l,N]=size(Test_array);

% Calculate the squared Eucleidian distance
for i=1:N
    for j=1:c
        distance(j)=sqrt((Test_array(:,i)-m_hat(:,j))'*(Test_array(:,i)-m_hat(:,j)));
    end
    [num,idx(i)]=min(distance);
end
classified=idx;