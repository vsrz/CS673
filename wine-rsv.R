

library(rminer) # load the rminer library

whitet <- read.table("winequality-white-t.csv",sep=";",header=TRUE)
redcomplete <- read.table("winequality-red.csv",sep=";",header=TRUE)
d <- redcomplete

v <- c("kfold", 5)

# 10 runs of 5-fold, multiple linear regression
MR <- mining(quality~., d, model="mr", Runs=10, method=v)
savemining(MR,"imr");
           
# Artificial Neural Network
m <- c(3, 100, "kfold", 4, "RAE")
s <- seq(1, 6, 1)
task <- "reg"
feature <- "RRSE"
#NN <- mining(quality~., d,model='mple',Runs=10,method=v,mpar=m,search=s,task=task,feature=feature, M='')
NN <- mining(quality~., d, Runs=10, method=c("kfold",3),model='dt')



# SVM
m <- c( NA, NA, "kfold", 4, "RAE")
s <- 2 ^ seq(-15, 3, 2)
