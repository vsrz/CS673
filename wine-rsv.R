library(rminer) # load the rminer library

#setseed(1337)

# 99 white wines testing set
whitetest <- read.table("winequality-white-t.csv",sep=";",header=TRUE)

# 4898 different white wines
whitecomplete <- read.table("winequality-white.csv",sep=';', header=TRUE)

# 1599 different red wines
redcomplete <- read.table("winequality-red.csv",sep=";",header=TRUE)

# heading labels for charts
heading <- c('fixed acidity',
              'volatile acidity',
              'citric acid',
              'residual sugar',
              'chlorides',
              'free sulfur dioxide',
              'total sulfur dioxide',
              'density',
              'pH',
              'sulphates',
              'alcohol',
              'quality')

# dataset to crunch on this run
#d <- whitetest
d <- whitecomplete
#d <- redcomplete

# common name for this run
#output_fn <- "whitetest"
output_fn <- "whitecomplete"
#output_fn <- "redcomplete"

# drop any wines that have any missing lab data
d <- na.omit(d)

# transform all data to a mean of 0 with 1 standard deviation
#colMeans(scaled.dat)
#apply(scaled.dat, 2, sd)
#scaled.dat
# no need to do this since rminer handles standard deviation for you

# 10 runs of 5-fold, multiple linear regression
v <- c("kfold", 5)
MR <- mining(quality~., d, model="mr", Runs=20, method=v)
savemining(MR, paste(output_fn, "-mr",sep=""));

# Artificial Neural Network
m <- c(3, 100, "kfold", 4, "RAE")
s <- seq(1, 6, 1)
NN=mining(quality~., d, model='mlpe', Runs=20, method=v, mpar=m, search=s, feat="s")
savemining(NN, paste(output_fn, "-nn",sep=""))

# ANN with 10 hidden nodes
ANN=mining(quality~.,d, model='mlp', search=10)
savemining(ANN, paste(output_fn, "-ann", sep=""))

# Support Vector Machine
# m <- (C, epsilon, vmethod, vpar, metric)
m <- c( NA, NA, "kfold", 4, "RAE")

# Sigma hyperparameter for SVM, S { 2^3, 2^2, ... 2^(-15) }
s <- 2 ^ seq(-15, 3, 2)

SV <- mining(quality~., d, model="svm", Runs=20, method=v, mpar=m, search=s, feat="s")
savemining(SV, paste(output_fn, "-svm",sep=""));

# reload test data so we dont have to crunch this again
datafiles <- "redcomplete"
#datafiles <- "whitetest"
#datafiles <- "whitecomplete"
SV <- loadmining(paste(datafiles, "-svm", sep=""))
NN <- loadmining(paste(datafiles, "-nn", sep=""))
ANN <- loadmining(paste(datafiles, "-ann", sep=""))
MR <- loadmining(paste(datafiles, "-mr", sep=""))

# output to console
sink()

# paired t-test and RAE mean and confidence intervals for SV:
print(output_fn)
print(t.test(mmetric(NN,metric="RAE"), mmetric(SV,metric="RAE"))))
print(meanint(mmetric(SV,metric="RAE")))

# accuracy curves
M <- vector('list',3)
M[[1]] <- SV
M[[2]] <- NN
M[[3]] <- MR

## show elapsed time
print("Avg Elapsed Time")
avtime_head <- c("SVM", "NN", "MR")
print(c(mean(SV$time),mean(c(NN$time)),mean(c(MR$time))))
names(avtime)
class(avtime)
avtime
summary(NN$mpar)
mgraph(M,graph="REC",leg=c("Support Vector Machine","Artificial Neural Network","Multiple Regression"), xval=15000)

# Analysis of input sensitivity analysis for SVM
mgraph(SV,graph="IMP",leg=heading, xval=0 )
mgraph(SV, graph="VEC", leg=heading, xval=11 )

# Analysis of input relativity for Multiple-regression
mgraph(MR,graph="IMP",leg=heading )
mgraph(MR, graph="VEC", leg=heading, xval=11)

########################################################
## Bar Plot Samples
bar <- table(d[,12])
barplot(bar, main="Red Wine", xlab="Sensory Preference", col="red")

alc <- table(d[,11])
barplot(alc, main="Red Wine", xlab="Alcohol Content", col="red")

# Sorted by column 11
dat <- dat[ order(dat[,11]),]
plot(x=dat[,11], y=dat[,12], main="Alcohol vs. Wine Sensory Preference", pch=1, xlab="Alcohol Content (% by volume)", ylab="Quality")
abline(lm(d[,12]~d[,11]), col="red")
lines(lowess(d[,11],d[,12]), col="blue")

sugar <- table(d[,4])
plot(x=d[,4], y=d[,12], main="Sugar vs. Wine Quality", pch=1)
abline(lm(d[,12]~d[,4]), col="red")
########################################################
##### fit a linear model
library(boot)
#model <- glm(quality~alcohol, data=d)
#MSE_LOOCV$delta[1]

MSE_LOOCV <- NULL

for (i in 1:11)
{
  model <- glm(quality~poly(alcohol, i), data=d)
  MSE_LOOCV[i] <- cv.glm(d, model)$delta[1]
}

##### k-fold CV
MSE_10KFV <- NULL

#repeat the process 9 parts training, 1 part testing
Kval <- 10
for (i in 1:11)
{
  model <- glm(quality~poly(alcohol, i), data=d)
  MSE_10KFV[i] <- cv.glm(d, model, K = Kval)$delta[1]
}
MSE_10KFV

###############################
##### Prediction test
M <- fit(quality~., whitetest, model="svm", search="heuristic")
P <- predict(M, whitetest)
print(mmetric(whitetest$quality, P, "AUC"))

data(sin1reg)
M=fit(y~., data=sin1reg, model="svm", search="heuristic")
P <- predict(M, sin1reg)
print(mmetric(sin1reg$y, P, "MAE"))
mgraph(sin1reg$y, P, graph="REC", Grid=10)
View(sin1reg)

### simple classification example.
data(iris)
M=fit(Species~.,iris,model="dt")
P=predict(M,iris)
print(mmetric(iris$Species,P,"CONF"))
print(mmetric(iris$Species,P,"ACC"))
print(mmetric(iris$Species,P,"AUC"))
print(mmetric(iris$Species,P,"ALL"))
mgraph(iris$Species,P,graph="ROC",TC=2,main="versicolor ROC",
       baseline=TRUE,leg="Versicolor",Grid=10)

### classification example with hyperparameter selection
# SVM
M=fit(Species~.,iris,model="svm",search=2^-3,mpar=c(3)) # C=3, gamma=2^-3
print(M@mpar) # gamma, C, epsilon (not used here)
M=fit(Species~.,iris,model="svm",search="heuristic10") # 10 grid search for gamma
print(M@mpar) # gamma, C, epsilon (not used here)
M=fit(Species~.,iris,model="svm",search="heuristic10") # 10 grid search for gamma
print(M@mpar) # gamma, C, epsilon (not used here)
M=fit(Species~.,iris,model="svm",search=2^seq(-15,3,2),
      mpar=c(NA,NA,"holdout",2/3,"AUC")) # same 0 grid search for gamma
print(M@mpar) # gamma, C, epsilon (not used here)
search=svmgrid(task="prob") # grid search as suggested by the libsvm authors
M=fit(Species~.,iris,model="svm",search=search) #
print(M@mpar) # gamma, C, epsilon (not used here)
M=fit(Species~.,iris,model="svm",search="UD") # 2 level 13 point uniform-design
print(M@mpar) # gamma, C, epsilon (not used here)
# MLPE
M=fit(Species~.,iris,model="mlpe",search="heuristic5") # 5 grid search for H
print(M@mpar)
M=fit(Species~.,iris,model="mlpe",search="heuristic5",
      mpar=c(3,100,"kfold",3,"AUC",2)) # 5 grid search for decay, inner 3-fold
print(M@mpar)

### regression example with mining
data(sin1reg)
M1=mining(y~.,sin1reg[,c(1,2,4)],model="mr",Runs=5)
M2=mining(y~.,sin1reg[,c(1,2,4)],model="mlpe",
          mpar=c(3,50),search=4,Runs=5,feature="simp")
L=vector("list",2); L[[1]]=M2; L[[2]]=M1
mgraph(L,graph="REC",xval=0.1,leg=c("mlpe","mr"),main="REC curve")

## 3rd example, use of naive method (most common class)
M=mining(Species~.,iris,Runs=1,method=c("kfold",3),model="naive")
print(mmetric(M,metric="CONF"))

data(iris)
M=fit(Species~.,iris,model="lr")
P=predict(M,iris)
print(mmetric(iris$Species,P,"CONF")) # confusion matrix


savemodel(SV, paste(datafiles,"svm","model", sep="-"))
savemodel(NN, paste(datafiles,"nn","model", sep="-"))
savemodel(MR, paste(datafiles,"mr","model", sep="-"))
MSV <- loadmodel(paste(datafiles,"svm","model", sep="-"))
MNN <- loadmodel(paste(datafiles,"nn","model", sep="-"))
MSV <- loadmodel(paste(datafiles,"mr","model", sep="-"))

M <- fit(quality~., MNN, model="mple")
P <- predict(redcomplete, NN)
print(mmetric(d, P, "CONF"))
M
PNN <- predict(NN, redcomplete)
