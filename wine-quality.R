# if needed, change the working directory with: getwd("YOUR-WORKING-DIRECTORY")
# warning: the whole execution of this R code requires some computation...
library(rminer) # load the rminer library

# Auxiliary function that performs the classification/regression predictive experiments 
# and writes the mining object for each model. Also, the test error average results 
# for each model are saved into label-metric.txt
# Several UCI examples of using this function are shown below.
datatest <- function(formula,data,label,models,Runs=10,v=c("kfold",5),task="default",feature="none",metric)
{
 cat("Mining dataset:",label,"with:",models,"and ",Runs,"runs:\n")
 
  if (label!="") 
  { 
    FILE=paste(label,"-",metric,".txt",sep=""); sink(FILE)} # all output is saved into FILE
    cat("dataset:",label,"metric:",metric,"\n")

    for(i in 1:length(models)) # cycle through all models (only "mlpe" and "svm" will work)
    {
      # search and mpar vectors  
      s <- NULL; 
      m <- NULL; 
      if(models[i]=="mlpe")
      { 
        m<-c(3,100,"kfold",3,metric) 
        s<-seq(0,9,1) # s=0,1,2,...,9
      }
      else if (models[i]=="svm")
      { 
        # NA = C or epsilon default heuristics 
        m <- c(NA,NA,"kfold",3,metric); 
        s<-2^seq(-15,3,2) # s=2^-15,2^-13,...,2^3
      }
      
      MM <- mining(formula,data,model=models[i],Runs=Runs,method=v,mpar=m,search=s,task=task,feature=feature)
      minesave <- paste(label,"-",models[i],"-",metric,sep="") # save mining results
      savemining(MM,minesave) 
      M <- mmetric(MM,metric=metric); MI=meanint(M) # output the average test error results
      cat("model: ",models[i]," ",metric,":",MI$mean,"+-",MI$int,"\n",sep="")
  }
 
  if(label!="") 
    sink() # restore output to the console
}

# white:
#w = read.table("winequality-white.csv",sep=";",header=TRUE)

#truncated white wine set
w <- read.table("winequality-white-t.csv",sep=";",header=TRUE)
datatest(quality~.,w,label="white",models=c("mlpe","svm"),task="reg",metric="RRSE")

#red
r <- read.table("winequality-red.csv", sep=";", header=TRUE)
