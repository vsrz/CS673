
library (rminer)

# 1599 different red wines
red_dataset <- read.table("winequality-red.csv",sep=";",header=TRUE)
#d <- red_dataset
#filename <- "RedWine"

# 4898 different white wines
white_dataset <- read.table("winequality-white.csv",sep=';', header=TRUE)
d <- white_dataset
filename <- "WhiteWine"

# 100 different white wines
short_dataset <- read.table("winequality-white-t.csv",sep=';', header=TRUE)
#d <- short_dataset
#filename <- "ShortWine"

################################################################################
# Data mining for each MR, NN, and SVM
#
method    <- c("kfold", 5)
model     <- "mr"
Runs      <- 20

# MR
MR        <- mining( quality~., d, model=model, Runs=Runs, method=method )

# NN
mpar      <- c( 3, 100, "kfold", 4, "RAE" )
search    <- seq( 1, 11, 1 )
model     <- "mple"
feat      <- "s"
NN        <- mining( quality~., d, model="mlpe", Runs=Runs, method=method, mpar=mpar, search=search, feat=feat )

# SVM
mpar      <- c( NA, NA, "kfold", 4, "RAE" )
SV        <- mining( quality~., d, model="svm", Runs=Runs, method=method, mpar=mpar, search=search, feat=feat )

################################################################################
# Save this mining session, so we don't have to do it again
#
savemining(NN, paste(filename, "nn", sep="-"))
savemining(SV, paste(filename, "sv", sep="-"))
savemining(MR, paste(filename, "mr", sep="-"))

################################################################################
##                                                                            ##
##                                                                            ##
##                      Post data generation, start here                      ##
##                                                                            ##
##                                                                            ##
################################################################################

# load previously mined datafiles
NN <- loadmining(paste(filename, "nn", sep="-"))
SV <- loadmining(paste(filename, "sv", sep="-"))
MR <- loadmining(paste(filename, "mr", sep="-"))

# define column headings
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

################################################################################
# Plot the average REC curves
#
M         <- vector( "list", 3 )
M[[1]]    <- SV
M[[2]]    <- MR
M[[3]]    <- NN
graph     <- "REC"
leg       <- c( "SVM", "NN", "MR" )
xval      <- 2
title     <- "Regressive Error Characteristic (REC) Curve"
mgraph( M, graph=graph, leg=leg, xval=xval, main=title)

################################################################################
# Input sensitivity analysis in %
#
graph     <- "IMP"
mgraph(SV,graph=graph,leg=heading, xval=0 )
mgraph(NN,graph=graph,leg=heading, xval=0 )


################################################################################
# Show human sensory preferences information chart
#
color     <- "#bb1111"
sensory_w <- table(white_dataset[,12])
sensory_r <- table(red_dataset[,12])
ylabel    <- "Number of wine samples"
xlabel    <- "Human Sensory Preference"
spacing   <- 0.01

barplot( sensory_w, main="White Wine", xlab="Sensory Preference", col=color, ylab=ylabel, space=spacing )
barplot( sensory_r, main="Red Wine", xlab="Sensory Preference", col=color, ylab=ylabel, space=spacing )

################################################################################
# Show the confusion matrix
#
M <- fit( quality~., d, model="lr")
P <- predict( M, d )
print( mmetric( d$quality, P, "CONF" ))
