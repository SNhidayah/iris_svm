# iris_svm
The following R commands are for multiclass classification problem using <i>one-vs-all</i> (OVA) strategy by default


<strong><h2> Classification of Iris species (<i>setosa</i>, <i>versicolor</i>, <i>verginica</i>) using Support Vector Machine </strong></h2>

In this session, the supervised classification technique of Support Vector Machine (SVM) was employed to classify Iris flowers based on their own species. Iris dataset was retrieved from https://archive.ics.uci.edu/ml/datasets/Iris . The 150 samples were splitted into 80% of training set and 20% of testing set. The 4 different kernels consisting of radial basis function (RBF), polynomial, linear and sigmoid were compared to find the best kernel that could be used to built final SVM model with an excellent accuracy.  


<h3> 1) Installing package and loading library </h3>

```bash
install.packages("e1071")

library(e1071)
```


<h3> 2) Read file </h3>

```bash
iris <- read.csv("iris_dataset.csv")
```


<h3> 3) Find the number of training set needed and set bootstrap to randomize all samples </h3>
In this analysis, 80% of the 150 samples was selected as training set. 

```bash
150*0.8

set.seed(1000)
```

<h3> 4) Randomly split the data into 80% of training set and 20% of testing set </h3>

```bash
train <- sample(1:150,120,replace = FALSE)
train_iris <- iris[train,]
test_iris <- iris[-train,]
```

<h3> 5) Kernel tuning </h3>
Instead of using default kernels settings, we tune kernel to find optimal values of gamma and cost parameters that gives the best model performance with less error. 
We use training set to build the SVM model.

```bash
tune_iris <- tune.svm(variety~., data=train_iris, gamma=seq(0.01,1,by=0.30),cost=seq(1,50,10)) 
```

<h3> 6) Overview the tuning </h3>

```bash
summary(tune_iris)
```


<h3> 7) Employ the best gamma and cost values  </h3>
Based on the tuning result, the optimal values of gamma = 0.01 and cost = 11 were selected to execute on different kernels. Different kernel will give different number of support vector machine. 

```bash
bestradial<- svm(variety~.,data = train_iris, cost=11, gamma=0.01)
bestpoly<- svm(variety~.,data = train_iris, kernel="polynomial", cost=11,gamma=0.01) 
bestlinear<- svm(variety~.,data = train_iris, kernel="linear", cost=11, gamma=0.01)
bestsigmoid<- svm(variety~.,data = train_iris, kernel="sigmoid",cost=11,gamma=0.01)
```


<h3> 8) Overview of all kernels based on the gamma and cost optimization </h3>

```bash
summary(bestradial)
summary(bestpoly)
summary(bestlinear)
summary(bestsigmoid)
```


<h3> 9) Accuracy of training set based on 10-fold cross validation </h3>

```bash
crossbestradial<- svm(variety~.,data = train_iris, cost=11, gamma=0.01,cross=10)
crossbestpoly<- svm(variety~.,data = train_iris, kernel="polynomial", cost=11,gamma=0.01,cross=10) 
crossbestlinear<- svm(variety~.,data = train_iris, kernel="linear", cost=11, gamma=0.01,cross=10)
crossbestsigmoid<- svm(variety~.,data = train_iris, kernel="sigmoid",cost=11,gamma=0.01,cross=10)
```

<h3> 10) Overview the training set accuracy of different kernels </h3>
The accuracy of different kernels was compared. Out of 4 kernels, RBF was observed to have an excellent accuracy of training set with 96.66667%

```bash
summary(crossbestradial)
summary(crossbestpoly)
summary(crossbestlinear)
summary(crossbestsigmoid)
```


<h3> 11) Prediction of the SVM model using testing set </h3>
The testing set was used to predict and validate the training set of SVM model

```bash
predictionbestradial<- predict(bestradial,test_iris)
predictionbestpoly<- predict(bestpoly,test_iris)
predictionbestlinear<- predict(bestlinear,test_iris)
predictionbestsigmoid<- predict(bestsigmoid,test_iris)
```

<h3> 12) Tabulate the confusion matrix using testing set </h3>
Confusion matrix was built to observe the accuracy of samples classification based on the testing set

```bash
table(predictionbestradial,test_iris$variety)
table(predictionbestpoly,test_iris$variety)
table(predictionbestlinear,test_iris$variety)
table(predictionbestsigmoid,test_iris$variety)
```

<h3> 13) Accuracy of testing set based on confusion matrix </h3>
From the overall kernels, RBF, linear and sigmoid gave the best accuracy with 0.9666667

```bash
sum(test_iris$variety == predictionbestradial)/30
sum(test_iris$variety == predictionbestpoly)/30
sum(test_iris$variety == predictionbestlinear)/30
sum(test_iris$variety == predictionbestsigmoid)/30
```


<h3> Conclusion </h3>
Although linear and sigmoid gives testing set with an accuracy of 96.66667%, however, both of these kernels possess slightly lower accuracy values in training set than RBF. Therefore, in this analysis, RBF kernel is the best kernel to be employed in building SVM model since this kernel gives the best accuracy values in training and testing set of iris data.



<h3> Reference: </h3>
The R commands were adapted and slightly modified based on Prof Vamsidhar Ambatipudi's Youtube https://youtu.be/jwY7pnBs1sI
