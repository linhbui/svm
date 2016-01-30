# Part 1
# Use svm to classify wine data
rm(list=ls())
library(e1071)
data <- read.csv('winequality-red.csv', header=TRUE, sep = ';')
# a. treat quality score as ordered factor
model1 <- svm(factor(quality, ordered = TRUE) ~ ., data = data, cross = 10)
summary(model1)
pred1 <- predict(model1, data)
error1 <- 1-sum(abs(data[,12] == pred1))/length(pred1)
error1 #[1] 0.3283302
# treat quality score as a numeric
model2 <- svm(quality ~ ., data = data, cross = 10)
summary(model2)
pred2 <- round(predict(model2, data))
error2 <- 1-sum(abs(data[,12] == pred2))/length(pred2)
error2 #[1] 0.3202001

# b. Plot with different pair of attributes
plot(model1, data, fixed.acidity ~ alcohol)
plot(model1, data, fixed.acidity ~ volatile.acidity)
plot(model1, data, total.sulfur.dioxide ~ pH)

# Part 2: Sonar data set with SVM
rm(list = ls())
oldpar <- par(no.readonly = TRUE)
par(mar = rep(1, 4))

library(e1071)
library(rpart)
library(MASS)

sonarTrain <- read.csv(file = "sonar_train.csv", header = FALSE)

sonarTrain$V61 <- as.factor(sonarTrain$V61)

model <- svm(V61 ~ ., data = sonarTrain)
summary(model)

x <- subset(sonarTrain, select = -V61)
y <- sonarTrain$V61
pred <- predict(model, x)
table(pred,y)
##       y
##pred -1  1
## -1 64  0
## 1   2 64

# Tuning with radial basis function kernel
obj <- tune(svm, V61 ~ ., data = sonarTrain, ranges = list(gamma = 2^(-1:1), cost = 2^(2:4)), tunecontrol = tune.control(sampling = "cross"))

summary(obj)
obj$best.parameters
##    gamma cost
##1   0.5    4

rbfModel <- svm(V61 ~ ., data = sonarTrain, gamma = 0.5, cost = 4)
summary(rbfModel)
rbfPred <- predict(rbfModel,x)
table(rbfPred, y)
##           y
##rbfPred -1  1
##     -1 66  0
##     1   0 64

# Tuning with linear kernel
C <- 2^(-10)
C <- 2 * C
C #[1] 0.001953125

linearModel1 <- svm(V61 ~ ., data = sonarTrain, kernel = "linear", cost = C)
summary(linearModel1)
linearPred1 <- predict(linearModel1, x)
table(linearPred1, y)
##              y
##linearPred -1  1
##        -1 47 13
##        1  19 51

# more laxed C
C <- 0.65

linearModel2 <- svm(V61 ~ ., data = sonarTrain, kernel = "linear", cost = C)
summary(linearModel2)
linearPred2 <- predict(linearModel2, x)
table(linearPred2, y)
##              y
##linearPred -1  1
##        -1 64  4
##        1  2  60

#b. Use RBF to classify the sonar test data
sonarTest <- read.csv(file = "sonar_train.csv", header = FALSE)

sonarTest$V61 <- as.factor(sonarTest$V61)
xTest <- subset(sonarTest,select=-V61)
yTest <- sonarTest$V61
predTest <- predict(rbfModel,xTest)
table(predTest, yTest)
##           yTest
## predTest -1  1
##       -1 66  0
##       1   0 64
testError <- 1-sum(abs(yTest == predTest))/length(predTest)
testError  #0

# Part 3: Glass dataset
# Random Forest
rm(list=ls())
library(randomForest)
install.packages('mlbench')
library(mlbench)
data(Glass, package = "mlbench")
str(Glass)

numrow <- 1:nrow(Glass)
set.seed(pi)
testnumrow <- sample(numrow, trunc(length(numrow)/3))
test <- Glass[testnumrow, ]
train <- Glass[-testnumrow, ]

x <- subset(train, select=-Type)
y <- train$Type
model <- randomForest(x,y)

xTest <- subset(test, select=-Type)
yTest <- test$Type
predictGlass <- predict(model, xTest)
error <- 1-sum(predictGlass == yTest)/length(yTest)
error # [1] 0.1549296

# Part 4: SVM on a new data set
rm(list=ls())
data <- read.csv('ProstateCancerDataESL.csv', header=TRUE, sep = ',')
head(data)
data$train <- as.factor(data$train)
summary(data)
## lcavol           lweight           age             lbph              svi        
## Min.   :-1.3471   Min.   :2.375   Min.   :41.00   Min.   :-1.3863   Min.   :0.0000  
## 1st Qu.: 0.5128   1st Qu.:3.376   1st Qu.:60.00   1st Qu.:-1.3863   1st Qu.:0.0000  
## Median : 1.4469   Median :3.623   Median :65.00   Median : 0.3001   Median :0.0000  
## Mean   : 1.3500   Mean   :3.629   Mean   :63.87   Mean   : 0.1004   Mean   :0.2165  
## 3rd Qu.: 2.1270   3rd Qu.:3.876   3rd Qu.:68.00   3rd Qu.: 1.5581   3rd Qu.:0.0000  
## Max.   : 3.8210   Max.   :4.780   Max.   :79.00   Max.   : 2.3263   Max.   :1.0000  
## lcp             gleason          pgg45             lpsa           train   
## Min.   :-1.3863   Min.   :6.000   Min.   :  0.00   Min.   :-0.4308   FALSE:30  
## 1st Qu.:-1.3863   1st Qu.:6.000   1st Qu.:  0.00   1st Qu.: 1.7317   TRUE :67  
## Median :-0.7985   Median :7.000   Median : 15.00   Median : 2.5915             
## Mean   :-0.1794   Mean   :6.753   Mean   : 24.38   Mean   : 2.4784             
## 3rd Qu.: 1.1787   3rd Qu.:7.000   3rd Qu.: 40.00   3rd Qu.: 3.0564             
## Max.   : 2.9042   Max.   :9.000   Max.   :100.00   Max.   : 5.5829

numrow <- 1:nrow(data)
set.seed(pi)
testnumrow <- sample(numrow, trunc(length(numrow)/3))
train <- data[-testnumrow, ]
x <- subset(train, select = -train)
y <- train$train

test <- data[testnumrow, ]
xTest <- subset(test, select = -train)
yTest <- test$train

model <- svm(train ~ ., data = train)
summary(model)

pred <- predict(model, x)
table(pred,y)
##              y
## pred    FALSE TRUE
## FALSE     0    0
## TRUE     20   45
# Looks pretty bad here....

predTest <- predict(model, xTest)
table(predTest, yTest)
##           yTest
## predTest FALSE TRUE
## FALSE     0    0
## TRUE     10   22

errorTest <- 1-sum(abs(yTest == predTest))/length(yTest)
errorTest #[1] 0.3125

# Tuning with radial basis function kernel
obj <- tune(svm, train ~ ., data = train, ranges = list(gamma = 2^(-1:1), cost = 2^(2:4)), tunecontrol = tune.control(sampling = "cross"))

summary(obj)
obj$best.parameters
##    gamma cost
##3    2    4

rbfModel <- svm(train ~ ., data = train, gamma = 2, cost = 4)
summary(rbfModel)
rbfPred <- predict(rbfModel,x)
table(rbfPred, y)
##             y
## rbfPred FALSE TRUE
## FALSE    20    0
## TRUE      0   45

rbfPredTest <- predict(rbfModel, xTest)
table(rbfPredTest, yTest)
##               yTest
## rbfPredTest FALSE TRUE
##    FALSE     0    1
##    TRUE     10   21
# Unfortunately this doesn't look that good either.
rbfErrorTest <- 1-sum(abs(yTest == rbfPredTest))/length(yTest)
rbfErrorTest # [1] 0.34375

# Tuning with linear kernel
C <- 2^(-10)
C <- 2 * C
C #[1] 0.001953125

linearModel1 <- svm(train ~ ., data = train, kernel = "linear", cost = C)
summary(linearModel1)
linearPred1 <- predict(linearModel1, x)
table(linearPred1, y)
##                 y
## linearPred1 FALSE TRUE
##    FALSE     0    0
##    TRUE     20   45

linearPredTest1 <- predict(linearModel1, xTest)
table(linearPredTest1, yTest)
##                    yTest
##  linearPredTest1 FALSE TRUE
##          FALSE     0    0
##          TRUE     10   22
# Sad.....
linearErrorTest1 <- 1-sum(abs(yTest == linearPredTest1))/length(yTest)
linearErrorTest1 # [1] 0.3125

# larger C
C <- 10

linearModel2 <- svm(train ~ ., data = train, kernel = "linear", cost = C)
summary(linearModel2)
linearPred2 <- predict(linearModel2, x)
table(linearPred2, y)
##                  y
## linearPred2 FALSE TRUE
##     FALSE     8    5
##     TRUE     12   40
linearPredTest2 <- predict(linearModel2, xTest)
table(linearPredTest2, yTest)
## linearPredTest2 FALSE TRUE
##         FALSE     3    5
##         TRUE      7   17

linearErrorTest2 <- 1-sum(abs(yTest == linearPredTest2))/length(yTest)
linearErrorTest2 # [1] 0.375

#b. Now use random forest to classify this data set
library(randomForest)
modelRF <- randomForest(x,y)

predictRF <- predict(modelRF, x)
table(y, predictRF)
##          predictRF
## y       FALSE TRUE
## FALSE    20    0
## TRUE      0   45
predictRFTest <- predict(modelRF, xTest)
table(yTest, predictRFTest)
##        predictRFTest
## yTest   FALSE TRUE
## FALSE     3    7
## TRUE      6   16

errorRFTest <- 1-sum(predictRFTest == yTest)/length(yTest)
errorRFTest # [1] 0.40625

# Part 5: SVM with kernel 'linear' to create regression prediction
rm(list=ls())
x <- seq(0.1, 5, by = 0.05) # the observed feature
y <- log(x) + rnorm(x, sd = 0.2) # the target for the observed feature
data <- as.data.frame(cbind(x, y))

model <- svm(y ~ ., data = data, kernel='linear')
predict <- predict(model, x)
error <- sqrt(sum((y-predict)^2))/length(y)
error #[1] 0.04573412

# tuning the model with kernel 'linear'
tunedObj <- tune(svm, y~., data = data, kernel="linear", ranges = list(gamma = seq(0,2,.5), cost = seq(1,3.5,0.5)), tunecontrol = tune.control(sampling = "cross"))
summary(tunedObj)
tunedObj$best.parameters
##     gamma cost
## 6     0  3

tunedModel <- svm(y ~ ., data = data, kernel = "linear", gamma=0, cost=3)
tunedPredict <- predict(tunedModel, x);
errorTuned <- sqrt(sum((y-tunedPredict)^2))/length(y)
errorTuned # [1] 0.04573413

# Try other kernels
# RDF - default
defaultModel <- svm(y ~ ., data = data)
defaultPredict <- predict(tunedModel, x);
defaultError <- sqrt(sum((y-defaultPredict)^2))/length(y)
defaultError # [1] 0.04573413

# RDF - tuned
tunedRDF <- tune(svm, y~., data = data, ranges = list(gamma = seq(0,2,.5), cost = seq(1,3.5,0.5)), tunecontrol = tune.control(sampling = "cross"))
summary(tunedRDF)
tunedRDF$best.parameters
##      gamma cost
## 30     2  3.5
tunedModelRDF <- svm(y ~ ., data = data, gamma = 2, cost = 1.5)
tunedRDFPredict <- predict(tunedModelRDF, x);
tunedRDFError <- sqrt(sum((y-tunedRDFPredict)^2))/length(y)
tunedRDFError #[1] 0.02214092

# Polynomial kernel
polyModel <- svm(y ~ ., data = data, kernel = 'polynomial')
polyPredict <- predict(polyModel, x);
polyError <- sqrt(sum((y-polyPredict)^2))/length(y)
polyError #[1] 0.04364514

# Tuning the polynomial kernel
tunedPoly <- tune(svm, y~., data = data, kernel = 'polynomial', ranges = list(gamma = seq(0,2,.5), cost = seq(1,3.5,0.5)), tunecontrol = tune.control(sampling = "cross"))
summary(tunedPoly)
tunedPoly$best.parameters
##      gamma cost
## 2     0.5   1
tunedModelPoly <- svm(y ~ ., data = data, kernel = 'polynomial', gamma = 0.5, cost = 1)
tunedPolyPredict <- predict(tunedModelPoly, x);
tunedPolyError <- sqrt(sum((y-tunedPolyPredict)^2))/length(y)
tunedPolyError #[1] 0.04299607

# Sigmoid kernel
sigmoidModel <- svm(y ~ ., data = data, kernel = 'sigmoid')
sigmoidPredict <- predict(sigmoidModel, x);
sigmoidError <- sqrt(sum((y-sigmoidPredict)^2))/length(y)
sigmoidError #[1] 0.3460107 

# Tuning the sigmoid kernel
tunedSigmoid <- tune(svm, y~., data = data, kernel = 'sigmoid', ranges = list(gamma = seq(0,2,.5), cost = seq(1,3.5,0.5)), tunecontrol = tune.control(sampling = "cross"))
summary(tunedSigmoid)
tunedSigmoid$best.parameters
##      gamma cost
## 1     0   1
tunedModelSigmoid <- svm(y ~ ., data = data, kernel = 'sigmoid', gamma = 0, cost = 1)
tunedSigmoidPredict <- predict(tunedModelSigmoid, x);
tunedSigmoidError <- sqrt(sum((y-tunedSigmoidPredict)^2))/length(y)
tunedSigmoidError #[1] 0.09264132

# Try adding parameter to the model
x2 <- x^2
newData <- as.data.frame(cbind(x, x2, y))

x <- newData[,1:2]
newModel <- svm(y ~ ., data = newData, kernel = 'linear')
summary(newModel)
# cost = 1, gamma = 0.5, epsilon = 0.1
newPredict <- predict(newModel, x)
newError <- sqrt(sum((y-newPredict)^2))/length(y)
newError #[1] 0.03073962

# trying ridge regression
library(ridge)
errRidge <- rep(0, 20)
lambdas <- rep(0, 20)
for (iLambda in seq(from = 0, to = 20)) { 
  exp <- (+2 -4*(iLambda/20))
  xlambda <- 10^exp
  modelRidge <- linearRidge(y~., data = newData,lambda=xlambda)
  predictedValues <- predict(modelRidge,x)
  errorRidge <- sqrt(sum((y-predictedValues)^2))/length(y)
  errRidge[iLambda] <- error
  lambdas[iLambda] <- xlambda
}

errRidge
plot(lambdas,errRidge)
minErrRidge <- min(errRidge)
minErrRidge #0.04573412
