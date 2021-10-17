library(caret)
library(naivebayes)
library(plyr)
library(dplyr)
library(arules)
library(arulesViz)


# Import 2nd bigger dataframe (portugese) and makes copy
df <- read.csv("student-por.csv")
originaldf <- df
# Merge Alcohol Columns
df$alcohol <- df$Walc + df$Dalc
df <- subset(df,select=c(-Dalc,-Walc))


# Overall alcohol consumption average 3.782743
mean(df$alcohol)

# Training and testing sets for NB and prediction
trainindex <- createDataPartition(df$alcohol,p=0.7,list=FALSE)
df$alcohol <- as.factor(df$alcohol)
trainset <- df[trainindex,]
testset <- df[-trainindex,]
nbtrain <- naive_bayes(alcohol ~.,data=trainset,laplace=1)
pred <- predict(nbtrain,testset)
confusionMatrix(pred,testset$alcohol)


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
# Filter by count and support
bettermaxalcrules <- maxalcrules[quality(maxalcrules)$count>3]
inspect(bettermaxalcrules) #29 rules but only counts of 2
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

