#assignment 2
#meghana gitay
#Importing data set universal bank csv file
View(UniversalBank)
#assigning colnames
colnames(UniversalBank)
summary(UniversalBank)
#getting rid of column names id and zip code
UniversalBank$ID = NULL
UniversalBank$ZIP.Code = NULL
summary(UniversalBank)
library(caret)
library(class)
library(ISLR)
library(dplyr)
library(readr)
library(gmodels)
library(FNN)

Model_range_normalized <- preProcess(UniversalBank,method = "range")
UniversalBank_norm <- predict(Model_range_normalized,UniversalBank)
summary(UniversalBank_norm)
View(UniversalBank_norm)

#Data Partition into testing and training sets
Train_index <- createDataPartition(UniversalBank$Personal.Loan, p = 0.6, list = FALSE)
train.df = UniversalBank_norm[Train_index,]
validation.df = UniversalBank_norm[-Train_index,]

#Question 1 (Perform k-nn classification with all the predictors expect id and zip code using k=1 )
To_Predict = data.frame(Age = 40, Experience = 10, Income = 84, Family = 2, CCAvg = 2, Education = 1, Mortgage = 0, Securities.Account = 0, CD.Account = 0, Online = 1, CreditCard = 1)
print(To_Predict)
Prediction <- knn(train = train.df[,1:7], test = To_Predict_norm[,1:7], cl = train.df$Personal.Loan, k = 1)
print(Prediction)

#Question 2 (reducing the effects of underfitting and overfitting)
set.seed(123)
UniversalBankcontrol <- trainControl(method = "repeatedcv", number = 3, repeats = 2)
searchGrid = expand.grid(k=1:10)

knn.model = train(Personal.Loan~., data = train.df, method = 'knn', tuneGrid = searchGrid, trControl = UniversalBankcontrol)
knn.model

#Question 3 (confusion matrix for the validation data that results from using the best k)
predictions <- predict(knn.model, validation.df)
confusionMatrix(predictions, validation.df$Personal.loan)

#Question 4 (classify the following customers)
To_Predict_norm = data.frame(Age = 40, Experience = 10, Income = 84, family = 2, CCAvg = 2, Education = 1, Mortgage = 0, Securities.Account = 0, CD.Account = 0, Online = 1, CreditCard = 1)
To_Predict_norm = predict(Model_range_normalized, To_Predict)
predict(knn.model, To_Predict_norm)

#Question 5 (confusion matrix of the test set with that of the training and validation sets)
train_size = 0.5
Train_index = createDataPartition(UniversalBank$Personal.Loan, p = 0.5, list = FALSE)
train.df = UniversalBank_norm[Train_index,]

test_size = 0.2
Test_index = createDataPartition(UniversalBank$Personal.Loan, p = 0.2, list = FALSE)
Test.df = UniversalBank_norm[Train_index,]

valid_size = 0.3
validation_index =  createDataPartition(UniversalBank$Personal.Loan, p = 0.3, list = FALSE)
validation.df = UniversalBank_norm[validation_index,]

Trainknn = knn(train=train.df[,-8], test = train.df[,-8], cl = train.df[,8], k =1)
Testknn <- knn(train = train.df[,-8], test = Test.df[,-8], cl = train.df[,8], k =1)
Validationknn <- knn(train = train.df[,-8], test = validation.df[,-8], cl = train.df[,8], k =1)

confusionMatrix(Trainknn, train.df[,8])
confusionMatrix(Testknn, Test.df[,8])  
confusionMatrix(Validationknn, validation.df[,8])

#conclusion comment: From the above matrices, we can see that the accuracies of Testing 
#and Training sets are exactly equal which means the algorithm is doing 
#what it is supposed to do that is avoiding overfitting or underfitting.
