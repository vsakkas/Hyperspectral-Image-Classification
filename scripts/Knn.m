function [z]=Knn(k,train_array,train_labels,array)

[l,N1]=size(train_array);
[l,N]=size(array);
num_of_classes=max(train_labels);
% Calculate the squared eucleidian distance of a point from each reference vector
for i=1:N
    distance=sum((array(:,i)*ones(1,N1)-train_array).^2);
    [sorted,nearest]=sort(distance);
    % Count occurence of each class for the top k reference vectors
    ref_vector=zeros(1,num_of_classes);
    for j=1:k
        class=train_labels(nearest(j));
        ref_vector(class)=ref_vector(class)+1;
    end
    [val,z(i)]=max(ref_vector);
end