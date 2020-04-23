library(rlang)
library(ggplot2)
library(Hmisc)
library(corrplot, quietly = TRUE)
library(igraph)
library(rpart, quietly = TRUE)
library(grid)
library(kernlab, quietly = TRUE)
library(nnet, quietly = TRUE)
library(sampling)
library(plyr)
library(dplyr)
library(rpart.plot)
library(caret)
library(xgboost)

########Load the dataset from file.##########
crs <- new.env()
fname <- "/Users/lince/Desktop/CA-DM/rawdata/csv_result-5year.csv"
crs$dataset <- read.csv(
  fname,
  na.strings = c(".", "NA", "", "?"),
  strip.white = TRUE,
  encoding = "UTF-8"
)

# The following variable selections have been noted.
crs$input     <-
  c(
    "Attr1",
    "Attr2",
    "Attr3",
    "Attr4",
    "Attr5",
    "Attr6",
    "Attr7",
    "Attr8",
    "Attr9",
    "Attr10",
    "Attr11",
    "Attr12",
    "Attr13",
    "Attr14",
    "Attr15",
    "Attr16",
    "Attr17",
    "Attr18",
    "Attr19",
    "Attr20",
    "Attr21",
    "Attr22",
    "Attr23",
    "Attr24",
    "Attr25",
    "Attr26",
    "Attr27",
    "Attr28",
    "Attr29",
    "Attr30",
    "Attr31",
    "Attr32",
    "Attr33",
    "Attr34",
    "Attr35",
    "Attr36",
    "Attr37",
    "Attr38",
    "Attr39",
    "Attr40",
    "Attr41",
    "Attr42",
    "Attr43",
    "Attr44",
    "Attr45",
    "Attr46",
    "Attr47",
    "Attr48",
    "Attr49",
    "Attr50",
    "Attr51",
    "Attr52",
    "Attr53",
    "Attr54",
    "Attr55",
    "Attr56",
    "Attr57",
    "Attr58",
    "Attr59",
    "Attr60",
    "Attr61",
    "Attr62",
    "Attr63",
    "Attr64"
  )

crs$numeric   <-
  c(
    "Attr1",
    "Attr2",
    "Attr3",
    "Attr4",
    "Attr5",
    "Attr6",
    "Attr7",
    "Attr8",
    "Attr9",
    "Attr10",
    "Attr11",
    "Attr12",
    "Attr13",
    "Attr14",
    "Attr15",
    "Attr16",
    "Attr17",
    "Attr18",
    "Attr19",
    "Attr20",
    "Attr21",
    "Attr22",
    "Attr23",
    "Attr24",
    "Attr25",
    "Attr26",
    "Attr27",
    "Attr28",
    "Attr29",
    "Attr30",
    "Attr31",
    "Attr32",
    "Attr33",
    "Attr34",
    "Attr35",
    "Attr36",
    "Attr37",
    "Attr38",
    "Attr39",
    "Attr40",
    "Attr41",
    "Attr42",
    "Attr43",
    "Attr44",
    "Attr45",
    "Attr46",
    "Attr47",
    "Attr48",
    "Attr49",
    "Attr50",
    "Attr51",
    "Attr52",
    "Attr53",
    "Attr54",
    "Attr55",
    "Attr56",
    "Attr57",
    "Attr58",
    "Attr59",
    "Attr60",
    "Attr61",
    "Attr62",
    "Attr63",
    "Attr64"
  )

crs$categoric <- NULL
crs$target    <- "class"
crs$risk      <- NULL
crs$ident     <- "id"
crs$ignore    <- NULL
crs$weights   <- NULL

# Backup before clearing outlier
crs$dataset.backup <- crs$dataset

########Define function##########

# Specify outlier
# dataset_org: dataset to be nullified (dataframe)
# lower_q: lower quantile (integer)
# upper_q: upper quantile (integer)
# threshold: range multiplier of difference between lower_q and upper_q to be considered threshold
nullify_outlier <-
  function(dataset_org, lower_q, upper_q, threshold) {
    # Exclude the 2 non-numerical columns
    dataset <- dataset_org[, -which(colnames(dataset_org) == 'id')]
    dataset <- dataset[, -which(colnames(dataset) == 'class')]
    
    dataset.outlier <- dataset_org
    
    # Create a variable to store the row id's to be removed
    Outliers <- c()
    for (i in colnames(dataset)) {
      # Empty Outliers
      Outliers <- c()
      
      Range = quantile(dataset[i], upper_q, na.rm = TRUE) -
        quantile(dataset[i], lower_q, na.rm = TRUE)
      
      # Get the Min/Max values
      max <-
        quantile(dataset[i], upper_q, na.rm = TRUE) + (Range * threshold)
      min <-
        quantile(dataset[i], lower_q, na.rm = TRUE) - (Range * threshold)
      
      # Get the id's using which
      idx <- which(dataset[i] < min | dataset[i] > max)
      
      # Append the outliers list
      Outliers <- c(Outliers, idx)
      
      # Sort, I think it's always good to do this
      Outliers <- sort(Outliers)
      
      # Print number of outliers
      print(paste(
        "Number of outliers in ",
        as.character(i),
        " is ",
        as.character(length(Outliers))
      ))
      # Set outlier variable to NA
      for (j in Outliers) {
        dataset.outlier[j, i] <- NA
      }
      
    }
    return(dataset.outlier)
  }


# Impute remaining missing values
# dataset_org: dataset to be nullified (dataframe)
impute_median <- function(dataset_org) {
  dataset <- dataset_org[,-which(colnames(dataset_org) == 'id')]
  dataset <- dataset[,-which(colnames(dataset) == 'class')]
  
  for (i in colnames(dataset)) {
    dataset_org[i] <-
      impute(dataset_org[i], median)
  }
  return(dataset_org)
}

# Data Normalisation
# dataset_org: dataset to be nullified (dataframe)
data_normalisation <- function(dataset_org) {
  dataset <- dataset_org[,-which(colnames(dataset_org) == 'id')]
  dataset <- dataset[,-which(colnames(dataset) == 'class')]
  
  for (i in colnames(dataset)) {
    dataset_org[i] <- c(scale(dataset_org[i]))
  }
  return(dataset_org)
}

# Storing train data mean and sd
storeTrain_normalisation <- function(dataset_org) {
  dataset <- dataset_org[, -which(colnames(dataset_org) == 'id')]
  dataset <- dataset[, -which(colnames(dataset) == 'class')]
  
  t <- c(colnames(dataset))
  t$mean <- apply(dataset, 2, mean)
  t$sd <- apply(dataset, 2, sd)
  t$median <- apply(dataset, 2, median)
  
  return(list(
    mean = t$mean,
    sd = t$sd,
    median = t$median
  ))
}

# Normalising testset with trainset's mean and sd
test_normalisation <- function(dataset_org, train_list) {
  dataset <- dataset_org[, -which(colnames(dataset_org) == 'id')]
  dataset <- dataset[, -which(colnames(dataset) == 'class')]
  
  for (i in colnames(dataset)) {
    dataset_org[i] <-
      (dataset_org[i] - train_list$mean[i]) / train_list$sd[i]
  }
  return(dataset_org)
}

impute_medianTrain <- function(dataset_org, train_list) {
  dataset <- dataset_org[, -which(colnames(dataset_org) == 'id')]
  dataset <- dataset[, -which(colnames(dataset) == 'class')]
  
  for (i in colnames(dataset)) {
    var <- train_list$median[i]
    dataset_org[i] <-
      impute(dataset_org[i], var)
  }
  return(dataset_org)
}

########Spliting training data and testing data##########

# Count total rows of 0 and 1
crs$count.zero <- sum(crs$dataset$class == '0')
crs$count.one <- sum(crs$dataset$class == '1')

# Random test data ID
mtx_test <-
  strata(
    crs$dataset,
    stratanames = "class",
    size = c(round(crs$count.zero * 0.3), round(crs$count.one * 0.3)),
    method = "srswor"
  )

crs$nobs  <- nrow(crs$dataset)
crs$test  <- mtx_test[, 2]

# Total dataset for training, before random
crs$train <- setdiff(seq_len(crs$nobs), crs$test)
# Leftover count of class 1
crs$count.one <- crs$count.one - round(crs$count.one * 0.3)
# Assembling new dataset with more one class
crs$dataset.moreone <- crs$dataset[crs$train,]

# Need dplyr package
crs$dataset.zero <- filter(crs$dataset.moreone, class == '0')
crs$dataset.one <- filter(crs$dataset.moreone, class == '1')
crs$dataset.one <- rbind(crs$dataset.one, crs$dataset.one)
crs$dataset.more <- rbind(crs$dataset.zero, crs$dataset.one)

mtx_train <-
  strata(
    crs$dataset.more,
    stratanames = "class",
    size = c(dim(crs$dataset.one)[1] * 4, dim(crs$dataset.one)[1]),
    method = "srswor"
  )
crs$train  <- mtx_train[, 2]

crs$dataset.test <- crs$dataset[crs$test, ]
crs$dataset.train <- crs$dataset.more[crs$train, ]

########Cleaning Data##########

# 1. Specifying outliers

## Specify outlier for 0 and 1
crs$dataset.zero <- filter(crs$dataset.train, class == '0')
crs$dataset.one <- filter(crs$dataset.train, class == '1')

crs$dataset.outlier_zero <-
  nullify_outlier(crs$dataset.zero, 0.25, 0.75, 1.5)
crs$dataset.outlier_one <-
  nullify_outlier(crs$dataset.one, 0.1, 0.9, 3)

crs$dataset.outlier <-
  rbind(crs$dataset.outlier_zero, crs$dataset.outlier_one)

# 2. Deleting missing value with NA > 20

# Counting missing value for every column
sapply(crs$dataset.outlier, function(x)
  sum(is.na(x)))

# Sum the number of missing value of every row
missingValues <- rowSums(is.na(crs$dataset.outlier))

# Find the rows with the most missing values
head(sort(missingValues, decreasing = TRUE), 100)

# Get the id's using which, the amount of missing value > 20
idx_mv <- which(missingValues > 20)
length(idx_mv)

# Delete missing value > 20
crs$dataset.missing <- crs$dataset.outlier[-idx_mv, ]

# 3. Add missing value with median

crs$dataset.addmissing <- crs$dataset.missing

# Find index of Attr59 with 0 value
idx_zero <- which(crs$dataset.missing['Attr59'] == 0)

# Compute standard deviation of Attr37
sd_37 = sd(crs$dataset.missing[['Attr37']], na.rm = TRUE)
mean_37 = mean(crs$dataset.missing[['Attr37']], na.rm = TRUE)

# Impute mediam + 4*sd of Attr37 with corresponding Attr59 with 0 value
for (i in idx_zero) {
  crs$dataset.addmissing[idx_zero, 'Attr37'] <-
    (mean_37 + 4 * sd_37)
}

crs$dataset.addmissing <- impute_median(crs$dataset.addmissing)

# Store train dataset mean and sd
crs$storeTrain <- storeTrain_normalisation(crs$dataset.addmissing)

# normalise testset
crs$dataset.testNorm <- crs$dataset.test

# Impute missing values in testset
# Find index of Attr59 with 0 value
idx_zero <- which(crs$dataset.testNorm['Attr59'] == 0)

# Impute mediam + 4*sd of Attr37 with corresponding Attr59 with 0 value
for (i in idx_zero) {
  crs$dataset.testNorm[idx_zero, 'Attr37'] <-
    (mean_37 + 4 * sd_37)
}

# Impute remaining NAs with trainset's median
crs$dataset.testNorm <-
  impute_medianTrain(crs$dataset.testNorm, crs$storeTrain)

# Then normalise testset based on trainset sd and mean
crs$dataset.testNorm <-
  test_normalisation(crs$dataset.testNorm, crs$storeTrain)


# 4. Data Normalisation (mean=0, sd=1)

crs$dataset.normalisation <-
  data_normalisation(crs$dataset.addmissing)

########Select Attrs from Correlation##########

crs$input     <-
  c(
    "Attr3",
    "Attr4",
    "Attr6",
    "Attr12",
    "Attr15",
    "Attr16",
    "Attr17",
    "Attr20",
    "Attr21",
    "Attr22",
    "Attr24",
    "Attr27",
    "Attr29",
    "Attr32",
    "Attr34",
    "Attr36",
    "Attr37",
    "Attr38",
    "Attr40",
    "Attr41",
    "Attr42",
    "Attr43",
    "Attr44",
    "Attr45",
    "Attr46",
    "Attr50",
    "Attr53",
    "Attr55",
    "Attr56",
    "Attr57",
    "Attr59",
    "Attr60",
    "Attr61",
    "Attr62",
    "Attr63",
    "Attr64"
  )

crs$numeric   <-
  c(
    "Attr3",
    "Attr4",
    "Attr6",
    "Attr12",
    "Attr15",
    "Attr16",
    "Attr17",
    "Attr20",
    "Attr21",
    "Attr22",
    "Attr24",
    "Attr27",
    "Attr29",
    "Attr32",
    "Attr34",
    "Attr36",
    "Attr37",
    "Attr38",
    "Attr40",
    "Attr41",
    "Attr42",
    "Attr43",
    "Attr44",
    "Attr45",
    "Attr46",
    "Attr50",
    "Attr53",
    "Attr55",
    "Attr56",
    "Attr57",
    "Attr59",
    "Attr60",
    "Attr61",
    "Attr62",
    "Attr63",
    "Attr64"
  )

crs$categoric <- NULL
crs$target    <- "class"
crs$risk      <- NULL
crs$ident     <- "id"
crs$ignore    <-
  c(
    "Attr1",
    "Attr2",
    "Attr5",
    "Attr7",
    "Attr8",
    "Attr9",
    "Attr10",
    "Attr11",
    "Attr13",
    "Attr14",
    "Attr18",
    "Attr19",
    "Attr23",
    "Attr25",
    "Attr26",
    "Attr28",
    "Attr30",
    "Attr31",
    "Attr33",
    "Attr35",
    "Attr39",
    "Attr47",
    "Attr48",
    "Attr49",
    "Attr51",
    "Attr52",
    "Attr54",
    "Attr58"
  )
crs$weights   <- NULL

########Cross Validation and Get Best Model##########

crs$dataset.cv_train <- crs$dataset.normalisation
crs$dataset.cv_test <- crs$dataset.testNorm

# Change the type of column Class to factor, otherwise it will throw error
crs$dataset.cv_train[, 'class'] <-
  as.factor(crs$dataset.cv_train[, 'class'])
levels(crs$dataset.cv_train$class) <-
  make.names(levels(crs$dataset.cv_train$class))

crs$dataset.cv_test[, 'class'] <-
  as.factor(crs$dataset.cv_test[, 'class'])
levels(crs$dataset.cv_test$class) <-
  make.names(levels(crs$dataset.cv_test$class))

# define fitControl to decide cv
fitControl <- trainControl(
  method = "cv",
  number = 10,
  # To save out of fold predictions for best parameter combinantions
  savePredictions = 'final',
  classProbs = T # To save the class probabilities of the out of fold predictions
  )

# 1. train first layer model

# first layer model: Knn
knnGrid <-  expand.grid(k = c(15))
crs$model_first_knn <-
  train(
    class ~ .,
    data = crs$dataset.cv_train[, c(crs$input, crs$target)],
    trControl = fitControl,
    method = "knn",
    tuneGrid = knnGrid
  )

# first layer model: Support vector machine
svmGrid <-  expand.grid(C = c(1))
crs$model_first_svm <-
  train(
    class ~ .,
    data = crs$dataset.cv_train[, c(crs$input, crs$target)],
    trControl = fitControl,
    method = "svmLinear",
    tuneGrid = svmGrid
  )

# first layer model: Generalized Linear Model
crs$model_first_lr <-
  train(
    class ~ .,
    data = crs$dataset.cv_train[, c(crs$input, crs$target)],
    trControl = fitControl,
    method = "glm"
  )

# first layer model: Neural Network
nnetGrid <-  expand.grid(size = c(2),decay = c(13))
crs$model_first_nnet <-
  train(
    class ~ .,
    data = crs$dataset.cv_train[, c(crs$input, crs$target)],
    trControl = fitControl,
    method = "nnet",
    tuneGrid = nnetGrid
  )

# first layer model: rpart
rpart2Grid <-  expand.grid(maxdepth = c(4))
crs$model_first_rpart2 <-
  train(
    class ~ .,
    data = crs$dataset.cv_train[, c(crs$input, crs$target)],
    trControl = fitControl,
    method = "rpart2",
    tuneGrid = rpart2Grid
  )

# Predicting the out of fold prediction probabilities for training data
crs$dataset.cv_train$oof_pred_knn <-
  crs$model_first_knn$pred$X1[order(crs$model_first_knn$pred$rowIndex)]
crs$dataset.cv_train$oof_pred_svm <-
  crs$model_first_svm$pred$X1[order(crs$model_first_svm$pred$rowIndex)]
crs$dataset.cv_train$oof_pred_lr <-
  crs$model_first_lr$pred$X1[order(crs$model_first_lr$pred$rowIndex)]
crs$dataset.cv_train$oof_pred_nnet <-
  crs$model_first_nnet$pred$X1[order(crs$model_first_nnet$pred$rowIndex)]
crs$dataset.cv_train$oof_pred_rpart2 <-
  crs$model_first_rpart2$pred$X1[order(crs$model_first_rpart2$pred$rowIndex)]

# Predicting probabilities for the test data
crs$dataset.cv_test$oof_pred_knn <-
  predict(crs$model_first_knn, crs$dataset.cv_test, type = 'prob')$X1
crs$dataset.cv_test$oof_pred_svm <-
  predict(crs$model_first_svm, crs$dataset.cv_test, type = 'prob')$X1
crs$dataset.cv_test$oof_pred_lr <-
  predict(crs$model_first_lr, crs$dataset.cv_test, type = 'prob')$X1
crs$dataset.cv_test$oof_pred_nnet <-
  predict(crs$model_first_nnet, crs$dataset.cv_test, type = 'prob')$X1
crs$dataset.cv_test$oof_pred_rpart2 <-
  predict(crs$model_first_rpart2, crs$dataset.cv_test, type = 'prob')$X1

# 2. train second layer model

# Predictors for top layer models
predictors_top <-
  c(
    'oof_pred_knn',
    'oof_pred_svm',
    'oof_pred_lr',
    'oof_pred_nnet',
    'oof_pred_rpart2'
  )

# second layer model: rpart2
rpart2GridSec <-  expand.grid(maxdepth = c(2))
crs$model_second_rpart2 <-
  train(
    crs$dataset.cv_train[, predictors_top],
    crs$dataset.cv_train[, 'class'],
    trControl = fitControl,
    method = "rpart2",
    tuneGrid = rpart2GridSec
  )

# 3.Predict using second layer model
crs$dataset.cv_test$rpart2_stacked <-
  predict(crs$model_second_rpart2, crs$dataset.cv_test[, predictors_top])

# First layer model predicting 0 and 1
crs$prediction_first_knn <-
  predict(crs$model_first_knn, newdata = crs$dataset.cv_test)
crs$prediction_first_svm <-
  predict(crs$model_first_svm, newdata = crs$dataset.cv_test)
crs$prediction_first_lr <-
  predict(crs$model_first_lr, newdata = crs$dataset.cv_test)
crs$prediction_first_nnet <-
  predict(crs$model_first_nnet, newdata = crs$dataset.cv_test)
crs$prediction_first_rpart2 <-
  predict(crs$model_first_rpart2, newdata = crs$dataset.cv_test)

# Second layer model predicting 0 and 1
crs$prediction_second_rpart2 <-
  predict(crs$model_second_rpart2, newdata = crs$dataset.cv_test)

# 4.ConfusionMatrix output

# First layer model testing data output
confusionMatrix(crs$prediction_first_knn, crs$dataset.cv_test$class, positive = "X1")
confusionMatrix(crs$prediction_first_lr, crs$dataset.cv_test$class, positive = "X1")
confusionMatrix(crs$prediction_first_nnet, crs$dataset.cv_test$class, positive = "X1")
confusionMatrix(crs$prediction_first_svm, crs$dataset.cv_test$class, positive = "X1")
confusionMatrix(crs$prediction_first_rpart2, crs$dataset.cv_test$class, positive = "X1")

# Second layer model testing data output
confusionMatrix(crs$prediction_second_rpart2, crs$dataset.cv_test$class, positive = "X1")

# First layer model training data output
confusionMatrix(crs$model_first_knn$pred$obs, crs$model_first_knn$pred$pred, positive = 'X1')
confusionMatrix(crs$model_first_lr$pred$obs, crs$model_first_lr$pred$pred, positive = 'X1')
confusionMatrix(crs$model_first_svm$pred$obs, crs$model_first_svm$pred$pred, positive = 'X1')
confusionMatrix(crs$model_first_nnet$pred$obs, crs$model_first_nnet$pred$pred, positive = 'X1')
confusionMatrix(crs$model_first_rpart2$pred$obs, crs$model_first_rpart2$pred$pred, positive = 'X1')