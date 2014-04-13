

library(rminer) # load the rminer library

setseed(1337)

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
#d <- whitecomplete
d <- redcomplete

# common name for this run
#output_fn <- "whitetest"
#output_fn <- "whitecomplete"
output_fn <- "redcomplete"

# drop any wines that have any missing lab data
d <- na.omit(d)

# 10 runs of 5-fold, multiple linear regression
v <- c("kfold", 5)
MR <- mining(quality~., d, model="mr", Runs=10, method=v)
savemining(MR, paste(output_fn, "-mr",sep=""));
           
# Artificial Neural Network
m <- c(3, 100, "kfold", 4, "RAE")
s <- seq(1, 6, 1)
#NN <- mining(quality~., d, Runs=10, method=c("kfold",3),model='dt')
NN=mining(quality~., d, model='mlpe', Runs=10, method=v, mpar=m, search=s, feat="s")
savemining(NN, paste(output_fn, "-ann",sep=""));

# Support Vector Machine
m <- c( NA, NA, "kfold", 4, "RAE")
s <- 2 ^ seq(-15, 3, 2)
SV <- mining(quality~., d, model="svm", Runs=10, method=v, mpar=m, search=s, feat="s")
savemining(MR, paste(output_fn, "-svm",sep=""));

# output to console
sink()

# paired t-test and RAE mean and confidence intervals for SV:
print(t.test(mmetric(NN,metric="RAE"), mmetric(SV,metric="RAE")))
print(meanint(mmetric(SV,metric="RAE")))

# accuracy curves
M <- vector('list',3)
M[[1]] <- SV
M[[2]] <- NN
M[[3]] <- MR

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
barplot(alc, main="Red Wine", xlab="Alcohol Content")

plot(x=d[,11], y=d[,12], main="Alcohol vs. Wine Sensory Preference", pch=1)
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
