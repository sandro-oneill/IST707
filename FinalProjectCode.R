#
# Authors: Rayanna Harduarsingh (rharduar) & Sandro O'Neill (aoneil02)
# IST 707 - Spring 2021
# Final Project - Student Alcohol Consumption
# 

#necessary packages
library(caret)
library(naivebayes)
library(plyr)
library(dplyr)
library(arules)
library(arulesViz)
library(tidyverse)
library(gridExtra)
library(factoextra)
library(cluster)
library(ggplot2)
library(stats)
library(kernlab)
library(e1071)
require(stringr)
require(randomForest)
library(rsample)
library(tidyverse)

######################
# Import and Pre-process
######################

# Import csv data into dataframe (portugese) 
df <- read.csv("student-por.csv")
# Merge Alcohol Columns and remove weekday/weekend variables
df$alcohol <- df$Walc + df$Dalc
df <- subset(df,select=c(-Dalc,-Walc))
originaldf <- df # Create copy for later reference


###########################
# Data Exploration
###########################
mean(df$G1) # 1st quarter avg grade 11.4
mean(df$G2) # 2nd quarter avg grade 11.6
mean(df$G3) # 3rd quarter avg grade 11.9
mean(df$absences) # Overall absence average 3.7

# Overall alcohol consumption average 3.782743 and histogram
mean(df$alcohol)
hist(df$alcohol,main="Histogram of Alcohol Consumption",
     xlab="2-10 Scale Alcohol Consumption",ylab = "Frequency")


###########################
# Logistic Regression
###########################
lrdata <- originaldf
# Convert characters to factors for logistic regression
lrdata <- lrdata %>%
  mutate_if(is.character,funs(as.factor))
mylogit <- glm(alcohol ~.,data=lrdata)
summary(mylogit)
#The from the output above we can see the variables that have a statistically significant
#Positive or negative influence on alcohol consumption
#Significant variables:
#sexM,famsizeLE3,Fjobservices,reasonother,studytime,nurseryyes,famrel,goout,health,absences

###########################
# Training and Testing Set creation
###########################

set.seed(123)
trainindex <- createDataPartition(df$alcohol,p=0.7,list=FALSE)
df$alcohol <- as.factor(df$alcohol)
trainset <- df[trainindex,]
testset <- df[-trainindex,]
trainset$alcohol <- as.factor(trainset$alcohol)
testset$alcohol <- as.factor(testset$alcohol)



#######################
# CLASSIFICATION ALGORITHMS
#######################
#####FINAL TUNED MODELS######

# Creates train_control variable for 3-fold cross validation when training models
train_control <- trainControl(
  method = "cv", 
  number = 3
)

###SVM###

#alcohol vs freetime, go out, absences, study time 
svm.model = svm(alcohol ~ freetime+goout+absences+studytime,
                 data = trainset,
                 type = "C",
                 kernel = "polynomial",
                 cross=3,
                 scale=FALSE)
summary(svm.model)
# Training Accuracy: 38.46%
pred.svm <- predict(svm.model, newdata = testset)
confusionMatrix(pred.svm, testset$alcohol)
#Accuracy: 33.51%


###NAIVE BAYES###

#set up tuning grid
search_grid <- expand.grid(usekernel = c(TRUE, FALSE),
                           laplace = c(0, 1), 
                           adjust = c(0,1,3,5,10))

nbm <- train(alcohol ~ freetime+goout+absences+studytime,
             data=trainset,
             method = "naive_bayes",
             trControl = train_control,
             tuneGrid = search_grid
)

#Verify the performance and test on testing data
# top 5 models
nbm$results %>% 
  top_n(5, wt = Accuracy) %>%
  arrange(desc(Accuracy))

# results for best model
confusionMatrix(nbm)

# Training Set Accuracy: 38.9%

#nb prediction
pred.nb <- predict(nbm, newdata = testset)
confusionMatrix(pred.nb, testset$alcohol)
# Test set accuracy 32.99%

###DECISION TREE###

# Decision tree tuning grid
search_grid <- expand.grid(.cp=c(0.01,0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45))
#alcohol vs all other variables
# train model
dt <- train(alcohol ~ freetime+goout+absences+studytime,
            data=trainset,
            method = "rpart",
            trControl = train_control,
            tuneGrid = search_grid
)

# top 5 modesl
dt$results %>% 
  top_n(5, wt = Accuracy) %>%
  arrange(desc(Accuracy))

# results for best model
confusionMatrix(dt)
# Training Set Accuracy 38.9%


#Prediction on Test set
pred.dt <- predict(dt, testset)
confusionMatrix(pred.dt, testset$alcohol)

# Testing Set Accuracy: 32.99%


###RANDOM FOREST###
# default RF model
rf <- randomForest(
  formula = alcohol ~ freetime+goout+absences+studytime,
  data = trainset,
  ntree = 500,
  mtry  = 2
)
#tuning
#alcohol vs all other variables
plot(rf)
search_grid = expand.grid(.mtry = (1:5))
rf = train(alcohol ~ freetime+goout+absences+studytime, 
           data = trainset, 
           method = "rf",
           metric = 'Accuracy',
           trControl = train_control,
           tuneGrid = search_grid)

# top 5 models
rf$results %>% 
  top_n(5, wt = Accuracy) %>%
  arrange(desc(Accuracy))

# results for best model
confusionMatrix(rf)

# Training SetAccuracy: 37.14%
pred.rf <- predict(rf, newdata = testset)
confusionMatrix(pred.rf, testset$alcohol)

# Testing Set Accuracy: 33.5%

###########################
# Association rule mining section
###########################

# Adds row number column
df$ID <- 1:nrow(df)
# Converts characters to factors and discretizises numbers
df <- df %>%
  select(-ID) %>%
  mutate_if(is.character,funs(as.factor)) %>%
  mutate_if(is.integer,funs(discretize))

# Counts of the different consumption factors
table(df$alcohol)

# Rules for students that drink the most (10)

maxalcrules <- apriori(df,parameter = list(supp = 0.001, conf = 0.9,maxlen = 5),appearance = list(default="lhs",rhs=c("alcohol=10","alcohol=9","alcohol=8","alcohol=7","alcohol=6","alcohol=5")))
inspect(maxalcrules) #3k rules

# Filter by count 
bettermaxalcrules <- maxalcrules[quality(maxalcrules)$count>3]
inspect(bettermaxalcrules) # Filtered down to 13 rules
# Rules for students that drink the least (2)

minalcrules <- apriori(df,parameter = list(supp = 0.001, conf = 0.9,maxlen = 5),appearance = list(default="lhs",rhs=c("alcohol=2","alcohol=3")))
inspect(minalcrules) # Too many rules

# Filter by count and support
betterminrules <- minalcrules[quality(minalcrules)$count>1 &quality(minalcrules)$support > 0.02]
inspect(betterminrules) #Leaves 26 rules

# Average number of absences 3.659476
mean(originaldf$absences)

# High absence rules

highabs <- apriori(df,parameter=list(supp=0.001,conf=0.9,maxlen=5),appearance=list(default="lhs",rhs="absences=[4,32]"))
inspect(highabs) #15k rules
betterhighabs <- highabs[quality(highabs)$count>10 & quality(highabs)$support > 0.02]
inspect(betterhighabs) #23 rules

# Low absences rules
lowabs <- apriori(df,parameter=list(supp=0.001,conf=0.9,maxlen=5),appearance=list(default="lhs",rhs="absences=[0,4)"))
inspect(lowabs)
betterlowabs <- lowabs[quality(lowabs)$count>10 & quality(lowabs)$support > 0.04]
inspect(betterlowabs) #18 high count rules


###########################
# K-means clustering section
###########################

numdata <- originaldf
# Subset so numdata only contains numeric variables
numdata <- subset(numdata,select=c(-school,-sex,-address,-famsize,-Pstatus,-Mjob,-Fjob,-reason,-guardian,-schoolsup,-famsup,-paid,-activities,-nursery,-higher,-internet,-romantic))
# Scales the numeric datasets
scaledk <- scale(numdata) 
# Determining optimal number of clusters
fviz_nbclust(scaledk,kmeans,method="wss") # elbow at clusters=2
km <- kmeans(scaledk,centers=2,nstart=25) #creates clusters 
fviz_cluster(km,data=numdata) #visualizes the 2 clusters
km$centers # Gives the averages for each clusters for the numeric variables
table(numdata$alcohol,km$cluster) # Gives alcohol frequency counts for each cluster
table(numdata$absences,km$cluster) # Gives absence frequency counts for each cluster
colMeans(numdata[km$cluster==1,]) #Gives the column averages for the numeric variables for cluster 1
colMeans(numdata[km$cluster==2,]) #Gives the column averages for the numeric variables for cluster 1
km$size # 1st cluster in 277, 2nd is 372






