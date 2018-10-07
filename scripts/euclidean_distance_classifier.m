function [classified]=euclidean_distance_classifier(m_hat, array)

[l,c]=size(m_hat);
[l,N]=size(array);

% Calculate the squared Eucleidian distance
for i=1:N
    for j=1:c
        distance(j)=sqrt((array(:,i)-m_hat(:,j))'*(array(:,i)-m_hat(:,j)));
    end
    [num,idx(i)]=min(distance);
end
classified=idx;