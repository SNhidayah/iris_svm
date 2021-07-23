# 1. Get the working directory
getwd()

# 2. View the directory
dir()

# 3. Read the file that we want
iris <- read.csv("iris_dataset.csv")


## Install SVM package(s); either caret package or e1071 package

# 4. Install package (e1071)
install.packages("e1071")

# 5. Read e1071 library
library(e1071)

# 6. Find the number of training set that we need. In this situation,80% (120/150 samples) was selected as training set
150*0.8

# 7. Set the bootstrap to randomize the 150 samples
set.seed(1000)

# 8. Randomly split the data into training and testing set
train <- sample(1:150,120,replace = FALSE)
train_iris <- iris[train,]
test_iris <- iris[-train,]


# 9. Instead of using default kernels settings, we tune SVM to find optimal values of gamma and cost parameters that gives best model performance with less error. We use training set to build the SVM model
tune_iris <- tune.svm(variety~., data=train_iris, gamma=seq(0.01,1,by=0.30),cost=seq(1,50,10)) 


# 10. Overview of the best performance with less error
summary(tune_iris)


# 11. Select the best gamma = 0.01 and cost = 11 values to execute SVM model using different kernels
bestradial<- svm(variety~.,data = train_iris, cost=11, gamma=0.01)
bestpoly<- svm(variety~.,data = train_iris, kernel="polynomial", cost=11,gamma=0.01) 
bestlinear<- svm(variety~.,data = train_iris, kernel="linear", cost=11, gamma=0.01)
bestsigmoid<- svm(variety~.,data = train_iris, kernel="sigmoid",cost=11,gamma=0.01)

# 12. Overview the summary details of all kernels
summary(bestradial)
summary(bestpoly)
summary(bestlinear)
summary(bestsigmoid)

# 13. Find the accuracy of different kernels using training set based on 10-fold cross validation
crossbestradial<- svm(variety~.,data = train_iris, cost=11, gamma=0.01,cross=10)
crossbestpoly<- svm(variety~.,data = train_iris, kernel="polynomial", cost=11,gamma=0.01,cross=10) 
crossbestlinear<- svm(variety~.,data = train_iris, kernel="linear", cost=11, gamma=0.01,cross=10)
crossbestsigmoid<- svm(variety~.,data = train_iris, kernel="sigmoid",cost=11,gamma=0.01,cross=10)

# 14. Overview the accuracy of different kernels using training set based on 10-fold cross validation
summary(crossbestradial)
summary(crossbestpoly)
summary(crossbestlinear)
summary(crossbestsigmoid)

# 15. Find the prediction of the models using testing set
predictionbestradial<- predict(bestradial,test_iris)
predictionbestpoly<- predict(bestpoly,test_iris)
predictionbestlinear<- predict(bestlinear,test_iris)
predictionbestsigmoid<- predict(bestsigmoid,test_iris)

# 16. Tabulate the confusion matrix using testing set
table(predictionbestradial,test_iris$variety)
table(predictionbestpoly,test_iris$variety)
table(predictionbestlinear,test_iris$variety)
table(predictionbestsigmoid,test_iris$variety)

# 17. Find the accuracy of different kernels using testing set 
sum(test_iris$variety == predictionbestradial)/30
sum(test_iris$variety == predictionbestpoly)/30
sum(test_iris$variety == predictionbestlinear)/30
sum(test_iris$variety == predictionbestsigmoid)/30
