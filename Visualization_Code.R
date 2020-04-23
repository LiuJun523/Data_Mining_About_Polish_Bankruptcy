########Analysizing Correlation#########

# 1. Calcuating correlation

# Correlations work for numeric variables only
crs$dataset.cor <-
  cor(crs$dataset.normalisation[, crs$numeric],
      use = "pairwise",
      method = "pearson")

# Find correlations between attributes/variables crs$cor[Attr,Attr]
# Order the correlations by their strength
crs$dataset.ord <- order(crs$dataset.cor[1,])
crs$dataset.cor <- crs$dataset.cor[crs$dataset.ord, crs$dataset.ord]

# Display the actual correlations
print(crs$dataset.cor)

# Graphically display the correlations
opar <- par(cex = 0.5)
corrplot(crs$dataset.cor, mar = c(0, 0, 1, 0))
title(main = "Correlation using Pearson",
      sub = paste("Rattle", format(Sys.time(), "%Y-%b-%d %H:%M:%S"), Sys.info()["user"]))
par(opar)

# Save correlation
write.table(
  crs$dataset.cor,
  file = "Dataset-after-correlation.csv",
  sep = ",",
  quote = FALSE,
  row.names = FALSE,
  col.names = TRUE
)


# 2. Grouping high correlation variables

var.corelation <- crs$dataset.cor

# Prevent duplicated pairs
var.corelation <- var.corelation * lower.tri(var.corelation)

# Set correlation threshold
corThreshold <- 0.8

# Group high positive correlation
check.corelation_pos <-
  which(var.corelation > corThreshold, arr.ind = TRUE)

graph.cor <-
  graph.data.frame(check.corelation_pos, directed = FALSE)
groups.cor <-
  split(unique(as.vector(check.corelation_pos)),
        clusters(graph.cor)$membership)
high_cor_positive <-
  lapply(
    groups.cor,
    FUN = function(list.cor) {
      rownames(var.corelation)[list.cor]
    }
  )

# Group high negative correlation
check.corelation_neg <-
  which(var.corelation < (-corThreshold), arr.ind = TRUE)

graph.cor <-
  graph.data.frame(check.corelation_neg, directed = FALSE)
groups.cor <-
  split(unique(as.vector(check.corelation_neg)),
        clusters(graph.cor)$membership)
high_cor_negative <-
  lapply(
    groups.cor,
    FUN = function(list.cor) {
      rownames(var.corelation)[list.cor]
    }
  )

# Print the results
print(paste("High +ve correlation > +", as.character(corThreshold)))
count_cor_positive <- 0
for (i in 1:length(high_cor_positive)) {
  print(paste("Group ", as.character(i), ": ", high_cor_positive[i]))
  
  count_cor_positive <-
    count_cor_positive + length(high_cor_positive[[i]])
}
count_pos_char <- as.character(as.integer(count_cor_positive))

print(paste(
  "Total of ",
  count_pos_char,
  " variables reduced to ",
  as.character(length(high_cor_positive)),
  " groups"
))

print(paste("High -ve correlation < -", as.character(corThreshold)))
count_cor_negative <- 0
for (i in 1:length(high_cor_negative)) {
  print(paste("Group ", as.character(i), ": ", high_cor_negative[i]))
  
  count_cor_negative <-
    count_cor_negative + length(high_cor_negative[[i]])
}
count_neg_char <- as.character(as.integer(count_cor_negative))

print(paste(
  "Total of ",
  count_neg_char,
  " variables reduced to ",
  as.character(length(high_cor_negative)),
  " groups"
))

########Var, Max, Min of CV of different models#########

# first layer
var_model<-c(var(crs$model_first_knn[["resample"]][["Accuracy"]]),
             var(crs$model_first_lr[["resample"]][["Accuracy"]]),
             var(crs$model_first_svm[["resample"]][["Accuracy"]]),
             var(crs$model_first_rpart2[["resample"]][["Accuracy"]]),
             var(crs$model_first_nnet[["resample"]][["Accuracy"]]))
mean_model<-c(mean(crs$model_first_knn[["resample"]][["Accuracy"]]),
              mean(crs$model_first_lr[["resample"]][["Accuracy"]]),
              mean(crs$model_first_svm[["resample"]][["Accuracy"]]),
              mean(crs$model_first_rpart2[["resample"]][["Accuracy"]]),
              mean(crs$model_first_nnet[["resample"]][["Accuracy"]]))
max_model<-c(max(crs$model_first_knn[["resample"]][["Accuracy"]]),
             max(crs$model_first_lr[["resample"]][["Accuracy"]]),
             max(crs$model_first_svm[["resample"]][["Accuracy"]]),
             max(crs$model_first_rpart2[["resample"]][["Accuracy"]]),
             max(crs$model_first_nnet[["resample"]][["Accuracy"]]))
min_model<-c(min(crs$model_first_knn[["resample"]][["Accuracy"]]),
             min(crs$model_first_lr[["resample"]][["Accuracy"]]),
             min(crs$model_first_svm[["resample"]][["Accuracy"]]),
             min(crs$model_first_rpart2[["resample"]][["Accuracy"]]),
             min(crs$model_first_nnet[["resample"]][["Accuracy"]]))
var_model
mean_model
max_model
min_model

# second layer
var(crs$model_second_rpart2[["resample"]][["Accuracy"]])
mean(crs$model_second_rpart2[["resample"]][["Accuracy"]])
max(crs$model_second_rpart2[["resample"]][["Accuracy"]])
min(crs$model_second_rpart2[["resample"]][["Accuracy"]])

########Predictors importance#########

gbmImp_knn <- varImp(crs$model_first_knn, scale = FALSE)
gbmImp_lr <- varImp(crs$model_first_lr, scale = FALSE)
gbmImp_svm <- varImp(crs$model_first_svm, scale = FALSE)
gbmImp_nnet <- varImp(crs$model_first_nnet, scale = FALSE)
gbmImp_rpart2 <- varImp(crs$model_first_rpart2, scale = FALSE)
plot(gbmImp_knn, top = 5, main="knn")
plot(gbmImp_lr, top = 5, main="lr")
plot(gbmImp_svm, top = 5, main="svm")
plot(gbmImp_nnet, top = 5, main="nnet")
plot(gbmImp_rpart2, top = 5, main="rpart2")

########Plot ROC#########

library(pROC)
rocknn <- plot.roc(crs$dataset.cv_test$class, 
                   predict(crs$model_first_knn, crs$dataset.cv_test, type = 'prob')$X1)
roclr <- plot.roc(crs$dataset.cv_test$class, 
                  predict(crs$model_first_lr, crs$dataset.cv_test, type = 'prob')$X1)
rocsvm <- plot.roc(crs$dataset.cv_test$class, 
                   predict(crs$model_first_svm, crs$dataset.cv_test, type = 'prob')$X1)
rocnnet <- plot.roc(crs$dataset.cv_test$class, 
                    predict(crs$model_first_nnet, crs$dataset.cv_test, type = 'prob')$X1)
rocrpart2 <- plot.roc(crs$dataset.cv_test$class, 
                      predict(crs$model_first_rpart2, crs$dataset.cv_test, type = 'prob')$X1)
rocrpart2Sec <- plot.roc(crs$dataset.cv_test$class, 
                      predict(crs$model_second_rpart2, crs$dataset.cv_test, type = 'prob')$X1)

plot(rocknn,
     type = "l", col="yellow",
     legacy.axes = TRUE,
     main="ROC"
)
par(new=TRUE) #continue plot
plot(roclr,
     type = "l", col ="red",
     legacy.axes = TRUE,
     axex = FALSE,
     xlab = "",
     ylab = "")
par(new=TRUE) #continue plot
plot(rocsvm,
     # , print.thres = c(.5), 
     type = "l", col ="black",
     legacy.axes = TRUE,
     axex = FALSE,
     xlab = "",
     ylab = "")
par(new=TRUE) #continue plot
plot(rocnnet,
     print.thres = c(.5),
     print.thres.pattern = "%.3f (Spec = %.2f, Sens = %.2f)",
     print.thres.cex = .8, 
     print.thres.col = "blue",
     type = "l", col ="blue",
     legacy.axes = TRUE,
     axex = FALSE,
     xlab = "",
     ylab = "")
par(new=TRUE) #continue plot
plot(rocrpart2,
     type = "l", col ="purple",
     legacy.axes = TRUE,
     axex = FALSE,
     xlab = "",
     ylab = "")
par(new=TRUE) #continue plot
plot(rocrpart2Sec,
     print.thres = c(.5),
     print.thres.pattern = "%.3f (Spec = %.2f, Sens = %.2f)",
     print.thres.cex = .8, 
     print.thres.col = "green",
     type = "l", col ="green",
     legacy.axes = TRUE,
     axex = FALSE,
     xlab = "",
     ylab = "")
par(new=TRUE) #continue plot
legend("bottomright", 
       legend = c("knn", "lr", "svm", "nnet","rpart2","rpart2Ens"), 
       col = c("yellow","red", "black","blue","purple","green"), 
       bty = "n", 
       lty = 1,
       pt.cex = 2, 
       cex = 1.2, 
       text.col = "black", 
       horiz = F , 
       inset = c(0.1, 0.1, 0.1, 0.1, 0.1,0.1))

########Base model tune (i.e. KNN)#########

train_test <- data.frame(
  id = c (1), 
  tr_a = c(2),
  tr_s = c(3),
  te_a = c(4),
  te_s = c(5),
  stringsAsFactors = FALSE
)

for(i in 1:30){
  knnGrid <-  expand.grid(k = c(30))
  crs$model_first <-
    train(
      class ~ .,
      data = crs$dataset.cv_train[, c(crs$input, crs$target)],
      trControl = fitControl,
      method = "knn",
      tuneGrid = knnGrid
    )
  
  CM_train <- confusionMatrix(crs$model_first$pred$obs, crs$model_first$pred$pred, positive = 'X1')
  CM_test <- confusionMatrix(predict(crs$model_first, crs$dataset.cv_test), crs$dataset.cv_test$class, positive = 'X1')
  
  # Create the second data frame
  newdata <- 	data.frame(
    id = c (i), 
    tr_a = c(CM_train$overall['Accuracy']),
    tr_s = c(CM_train$byClass['Sensitivity']),
    te_a = c(CM_test$overall['Accuracy']),
    te_s = c(CM_test$byClass['Sensitivity']),
    stringsAsFactors = FALSE
  )
  
  train_test <- rbind(train_test,newdata)
}

########Second model tune (i.e. rPart2)#########

train_test <- data.frame(
  id = c (1), 
  tr_a = c(2),
  tr_s = c(3),
  te_a = c(4),
  te_s = c(5),
  stringsAsFactors = FALSE
)
for(i in 1:20){
  rpart2GridSec <-  expand.grid(maxdepth = c(i))
  crs$model_second <-
    train(
      crs$dataset.cv_train[, predictors_top],
      crs$dataset.cv_train[, 'class'],
      trControl = fitControl,
      method = "rpart2",
      tuneGrid = rpart2GridSec
    )
  
  CM_train <- confusionMatrix(crs$model_second$pred$obs, crs$model_second$pred$pred, positive = 'X1')
  CM_test <- confusionMatrix(predict(crs$model_second, crs$dataset.cv_test), crs$dataset.cv_test$class, positive = 'X1')
  
  # Create the second data frame
  newdata <- 	data.frame(
    id = c (i), 
    tr_a = c(CM_train$overall['Accuracy']),
    tr_s = c(CM_train$byClass['Sensitivity']),
    te_a = c(CM_test$overall['Accuracy']),
    te_s = c(CM_test$byClass['Sensitivity']),
    stringsAsFactors = FALSE
  )
  
  train_test <- rbind(train_test,newdata)
}

########Correlation of Second Layer#########

crs$dataset.cor <- cor(crs$dataset.cv_train[, predictors_top], use = "pairwise", method = "pearson")

# Find correlations between attributes/variables crs$cor[Attr,Attr]
# Order the correlations by their strength
crs$dataset.ord <- order(crs$dataset.cor[1,])
crs$dataset.cor <- crs$dataset.cor[crs$dataset.ord, crs$dataset.ord]

# Display the actual correlations
print(crs$dataset.cor)

# Graphically display the correlations
opar <- par(cex = 0.5)
corrplot(crs$dataset.cor, mar = c(0, 0, 1, 0))
title(main = "Correlation using Pearson",
      sub = paste("Rattle", format(Sys.time(), "%Y-%b-%d %H:%M:%S"), Sys.info()["user"]))
par(opar)

########ROC for individual models (i.e. rpart2)#########
rpart2Grid <-  expand.grid(maxdepth = c(3))
crs$model_roc1 <-
  train(
    class ~ .,
    data = crs$dataset.cv_train[, c(crs$input, crs$target)],
    trControl = fitControl,
    method = "rpart2",
    tuneGrid = rpart2Grid
  )
rpart2Grid <-  expand.grid(maxdepth = c(4))
crs$model_roc2 <-
  train(
    class ~ .,
    data = crs$dataset.cv_train[, c(crs$input, crs$target)],
    trControl = fitControl,
    method = "rpart2",
    tuneGrid = rpart2Grid
  )
rpart2Grid <-  expand.grid(maxdepth = c(5))
crs$model_roc3 <-
  train(
    class ~ .,
    data = crs$dataset.cv_train[, c(crs$input, crs$target)],
    trControl = fitControl,
    method = "rpart2",
    tuneGrid = rpart2Grid
  )
rpart2Grid <-  expand.grid(maxdepth = c(6))
crs$model_roc4 <-
  train(
    class ~ .,
    data = crs$dataset.cv_train[, c(crs$input, crs$target)],
    trControl = fitControl,
    method = "rpart2",
    tuneGrid = rpart2Grid
  )

roc1 <- plot.roc(crs$dataset.cv_test$class, 
                   predict(crs$model_roc1, crs$dataset.cv_test, type = 'prob')$X1)
roc2 <- plot.roc(crs$dataset.cv_test$class, 
                  predict(crs$model_roc2, crs$dataset.cv_test, type = 'prob')$X1)
roc3 <- plot.roc(crs$dataset.cv_test$class, 
                   predict(crs$model_roc3, crs$dataset.cv_test, type = 'prob')$X1)
roc4 <- plot.roc(crs$dataset.cv_test$class, 
                    predict(crs$model_roc4, crs$dataset.cv_test, type = 'prob')$X1)

plot(roc1,
     print.thres = c(.5),
     print.thres.pattern = "%.3f (Spec = %.2f, Sens = %.2f)",
     print.thres.cex = .8, 
     print.thres.col = "red",
     type = "l", col="red",
     legacy.axes = TRUE,
     main="ROC-RPATR2"
)
par(new=TRUE) #continue plot
plot(roc2,
     print.thres = c(.5),
     print.thres.pattern = "%.3f (Spec = %.2f, Sens = %.2f)",
     print.thres.cex = .8, 
     print.thres.col = "blue",
     type = "l", col ="blue",
     legacy.axes = TRUE,
     axex = FALSE,
     xlab = "",
     ylab = "")
par(new=TRUE) #continue plot
plot(roc3,
     print.thres = c(.5),
     print.thres.pattern = "%.3f (Spec = %.2f, Sens = %.2f)",
     print.thres.cex = .8, 
     print.thres.col = "green",
     type = "l", col ="green",
     legacy.axes = TRUE,
     axex = FALSE,
     xlab = "",
     ylab = "")
par(new=TRUE) #continue plot
plot(roc4,
     print.thres = c(.5),
     print.thres.pattern = "%.3f (Spec = %.2f, Sens = %.2f)",
     print.thres.cex = .8, 
     print.thres.col = "black",
     type = "l", col ="black",
     legacy.axes = TRUE,
     axex = FALSE,
     xlab = "",
     ylab = "")
par(new=TRUE) #continue plot
legend("bottomright", 
       legend = c("3", "4", "5", "6"), 
       col = c("red","blue", "green","black"), 
       bty = "n", 
       lty = 1,
       pt.cex = 2, 
       cex = 1.2, 
       text.col = "black", 
       horiz = F , 
       inset = c(0.1, 0.1, 0.1, 0.1))

########Save to csv#########
write.table(
  crs$dataset.normalisation,
  file = "Dataset-Processed-Train.csv",
  sep = ",",
  quote = FALSE,
  row.names = FALSE,
  col.names = TRUE
)
write.table(
  crs$dataset.testNorm,
  file = "Dataset-Processed-Test.csv",
  sep = ",",
  quote = FALSE,
  row.names = FALSE,
  col.names = TRUE
)