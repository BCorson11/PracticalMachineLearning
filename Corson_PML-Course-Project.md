---
title: "Practical Machine Learning Assignment"
author: "BCorson11"
date: "3/25/2022"
output:
  html_document:
    keep_md: yes
    pdf_document: default
  pdf_document: default
---



# Executive Summary

One thing that people regularly do is quantify how  much of a particular activity they do, but they rarely quantify how well they do it.  Below is analysis on predicting how an individual performed an activity (correctly or incorrectly).


# Data Loading

Below is the code to load the required packages and to load the test and train data.  I then selected certain variables from the list of 160 that had data inputted.  You can see that I only selected 53 variables because most of the columns in the dataset were empty or very sparsely populated.  The last bit of code sets the parameters for the Cross Validation I did in my model creation.  I decided to perform a K-Folds Cross-Validation with 10 folds.


```r
setwd("C:/Users/Corson/Desktop/Coursera/PML")
library(caret)
library(kernlab)
library(dplyr)
library(tidyverse)
library(earth)

test <- read.csv("pml-testing.csv")
train <- read.csv("pml-training.csv")
train$classe <- as.factor(train$classe)

seltrain <- train %>% select(starts_with(c("Roll", "Pitch", "Yaw", "Total_Acc",
                                           "gyros", "accel", "magnet", "classe"))) 
seltest <- test %>% select(starts_with(c("Roll", "Pitch", "Yaw", "Total_Acc",
                                           "gyros", "accel", "magnet", "problem_id")))

##Building control and validation sets for K-Fold Cross-Validation.
set.seed(125)
train_control <- trainControl(method = "cv",
                              number = 10)
```

# Model Creation

Below is the code I went through to build different models.  

I started with a tree models.  I fit one tree model with no pre-processing and no specified Cross Validation.  I then fit a tree model on the principal components and the last tree model I fit was on the principle components with K-Fold Cross Validation with 10 folds.

I then tried boosted models.  I fit 4 models using the "gbm" method.  The first was on the original data with no Cross Validation.  The second was on the principal components.  The third included K-Fold Cross Validation with 10 folds and the last was fit to the principal components and it including the K-Fold Cross Validation.

I've printed the accuracies from all the different models I fit.  You can see that the highest accuracy was the boosted model including the K-Fold Cross Validation.


```r
preProc <- preProcess(seltrain[,-53]+1, method = "pca")
trainPC <- predict(preProc, seltrain[,-53]+1)
trainPC$classe <- seltrain$classe

modelFitTree <- train(classe ~., data = seltrain, method = "rpart")
modelFitTreeCV <- train(classe ~., data = seltrain, method = "rpart", trControl = train_control)
modelFitTreePC_CV <- train(classe ~., data = trainPC, method = "rpart", trControl = train_control)
```


```r
modelFitBoost <- train(classe ~ ., method = "gbm", data = seltrain, verbose = FALSE)
```


```r
modelFitPCBoost <- train(classe ~ ., method = "gbm", data = trainPC, verbose = FALSE)
```


```r
modelFitBoostCV <- train(classe ~., data = seltrain, method = "gbm", verbose = FALSE,
                           trControl = train_control)
```


```r
modelFitPCBoostCV <- train(classe ~., data = trainPC, method = "gbm", verbose = FALSE,
                           trControl = train_control)
```


```r
accuracy_list <- c(max(modelFitTree$resample$Accuracy), 
                   max(modelFitTreeCV$resample$Accuracy), max(modelFitTreePC_CV$resample$Accuracy), 
                   max(modelFitBoost$resample$Accuracy), max(modelFitPCBoost$resample$Accuracy),
                   max(modelFitBoostCV$resample$Accuracy), max(modelFitPCBoostCV$resample$Accuracy))

names(accuracy_list) <- c("Tree Model", "Tree_CV", "Tree_PC_CV", "Boost Model", "Boost_PC", "Boost_CV", "Boost_PC_CV")
print(accuracy_list)
```

```
##  Tree Model     Tree_CV  Tree_PC_CV Boost Model    Boost_PC    Boost_CV 
##   0.6004725   0.5249745   0.4199796   0.9665874   0.8235129   0.9658512 
## Boost_PC_CV 
##   0.8343527
```

```r
Boost_CV_Acc <- max(modelFitBoostCV$resample$Accuracy)
```

# Out of Sample Error

I know that my out of sample error rate is going to be higher than my in sample error rate.  Therefore, I know my accuracy for the test data will be less than 0.9658512.  Since I used K-Fold CV with 10 Folds I do expect the accuracy to be close to the accuracy on data used to build the model.  
