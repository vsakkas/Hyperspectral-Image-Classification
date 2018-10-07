# Hyperspectral Image Classification (HSI)

This repository contains the solution for the project of the Pattern Recognition - Machine Learning class of the [Department of Informatics and Telecommunications of UoA](http://www.di.uoa.gr/eng). Purpose of the project was to classify the pixels of a provided hyperstrectral image into 5 different categories with the help of three different supervised classifiers.

The classifiers that were implemented in order to classify the pixels of the image are the following:
* Naive Bayes
* Minimum Eucleidian Distance
* K-Nearest Neighbor (KNN)

The script is split into 2 phases. First, the dataset is loaded and split into training an testing sets. As their names imply, they are used to train and then evaluate the performance of the 3 classifiers. In order to evaluate how accurate each classifier is, we calculate the error rate, a confusion matrix and then use the confusion matrix to find the precision of the trained classifier.

Training the KNN classifier has one more step compared to the other 2 classifiers. Before we train KNN using all of the training dataset, we initially use 5-fold cross validation to find the best value for K. More specifically, we split the training dataset into 5 subsets and use them to train 5 different KNN classifiers and find their average error rate. This process is done for all values of K from 1 up to 17 by skipping every second value (1->3->5-> ... and so on). We eventually keep the value of K that gave us the lowest average error rate and train the KNN classifier using the whole training dataset.

The second phase of the script uses again all 3 classifiers, but this time we classify the whole dataset (both training and testing datasets). Afterwards, we display the classified pixels which can be seen below:


| Naive Bayes | Eucleidian Distance | KNN | Ground Truth |
| :--------:|:---------:|:--------:|:--------:|
| ![alt tag](https://github.com/vsakkas/Hyperspectral-Image-Classification/blob/master/naive_bayes.jpg)   | ![alt tag](https://github.com/vsakkas/Hyperspectral-Image-Classification/blob/master/eucleidian.jpg)    | ![alt tag](https://github.com/vsakkas/Hyperspectral-Image-Classification/blob/master/knn.jpg)  | ![alt tag](https://github.com/vsakkas/Hyperspectral-Image-Classification/blob/master/ground_truth.jpg) |

As it can be seen from the above images, KNN has succesfully classified all pixels correctly. More specifically, by running the ```hsi.m``` script we get the following error rates:

| Naive Bayes | Eucleidian Distance | KNN |
| :----------:|:---------:|:--------:|
|   0.0259%   |  0.0245%  |   0.0%   |

By looking at the images, it's easy to see that there is a an empty space between the pixels that belong to different classes. This makes the KNN classifier an ideal choice which is also why it provides 100% accuracy in this test. On the other hand, the Naive Bayes and the Eucleidian classifiers rely on calculating the Likelihood parameters which require either a high number of points in our dataset or a low number oh dimensions (or both, preferrably). Unfortunately, this is not the case in this test, which is why those 2 classifiers fail to correctly classify all pixels. As such, the best out of these 2 classifiers for this test is the KNN classifier.

## Dataset
The dataset was provided by the Pattern Recognition - Machine Learning class of the [Department of Informatics and Telecommunications of UoA](http://www.di.uoa.gr/eng). It represents one area of the Salinas Valley located in California. The provided hyperspectral image has a resolution of 150x150 pixels and contains 204 spectral bands (ranging from 0.2μm to 2.4μm) and a spatial resolution of 3.7 meters (meaning that the hyperstrectral image can be represented by a 150x150x204 cube).

## Getting Started

To get the code up and running on your local machine, simply follow the following instructions.

### Prerequisites

First, you need to have MATLAB installed on your local system. This project was tested on version 2017a, but other versions of the software environment should be compatible.

### Downloading

Get a copy of the project
```
git clone https://github.com/vsakkas/Hyperspectral-Image-Classification.git
```
And to enter the directory of the downloaded project, simply type:
```
cd Hyperspectral-Image-Classification
```
### Running

To run the provided code, open MATLAB, navigate to the ```scripts``` directory, run the ```hsi.m``` file.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details
