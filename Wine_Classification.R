
# Loading required R packages

library(tidyverse)  # for easy data manipulation and visualisation
library(caret)  # for easy machine learning workflow

# Reading the data

wine<-read.table("F:/DSP Projects/Project 4/Dataset/wine.data.txt", sep="\t")
View(vino)
vino<-read.delim("F:/DSP Projects/Project 4/Dataset/wine.data.txt", header = FALSE, sep=",")

# Labeling the input dataset

library(dplyr)
vino<-data.frame(vino)
oldnames<-c("V1","V2","V3","V4","V5","V6","V7","V8","V9","V10","V11","V12","V13","V14")
newnames<-c("Type","Alcohol","Malic Acid","Ash","Alcalinity of Ash","Magnesium","Total phenols","Flavonoids","Non flavonoid phenols","Proanthocyanins","Color Intensity","Hue","Dilution","Proline")
vino <- vino %>% rename_at(vars(oldnames), ~ newnames)

# Preliminary analysis

dim(vino)
str(vino)
summary(vino)
table(vino$Type)

# Exploratory analysis

# Histogram

hist(vino$Alcohol, col='red')
hist(vino$`Malic Acid`, col='red')
hist(vino$Ash, col='red')
hist(vino$`Alcalinity of Ash`, col='red')
hist(vino$Magnesium, col='red')
hist(vino$`Total phenols`, col='red')
hist(vino$Flavonoids, col='red')
hist(vino$`Non flavonoid phenols`, col='red')
hist(vino$Proanthocyanins, col='red')
hist(vino$`Color Intensity`, col='red')
hist(vino$Hue, col='red')
hist(vino$Dilution, col='red')
hist(vino$Proline, col='red')

# Box plot

boxplot(vino$Alcohol, horizontal = TRUE, col='red', main="Box Plot of Alcohol")
boxplot(vino$`Malic Acid`, horizontal = TRUE, col='red', main="Box Plot of Malic Acid")
boxplot(vino$Ash, horizontal = TRUE, col='red', main="Box Plot of Ash")
boxplot(vino$`Alcalinity of Ash`, horizontal = TRUE, col='red', main="Box Plot of Alcalinity of Ash")
boxplot(vino$Magnesium, horizontal = TRUE, col='red', main="Box Plot of Magnesium")
boxplot(vino$`Total phenols`, horizontal = TRUE, col='red', main="Box Plot of Total phenols")
boxplot(vino$Flavonoids, horizontal = TRUE, col='red', main="Box Plot of Flavoids")
boxplot(vino$`Non flavonoid phenols`, horizontal = TRUE, col='red', main="Box Plot of Non flavonoid phenols")
boxplot(vino$Proanthocyanins, horizontal = TRUE, col='red', main="Box Plot of Proanthocyanins")
boxplot(vino$`Color Intensity`, horizontal = TRUE, col='red', main="Box Plot of Color Intensity")
boxplot(vino$Hue, horizontal = TRUE, col='red', main="Box Plot of Hue")
boxplot(vino$Dilution, horizontal = TRUE, col='red', main="Box Plot of Dilution")
boxplot(vino$Proline, horizontal = TRUE, col='red', main="Box Plot of Proline")

# Bar plot
barplot(table(vino$Type), main="Bar plot of wine types", xlab="Wine Type", ylab="Count", border="yellow", col="green")

# Splitting data into training and test data sets

tr=seq(round(0.75*length(vino$Type)))
Y_train=vino[tr,1]
Y_test=vino[-tr,1]
X_train=vino[tr,-1]
X_test=vino[-tr,-1]

# Estimate preprocessing parameters

preproc.param <- X_train %>% preProcess(method = c("center", "scale"))

# Transform the data using the estimated parameters

X_train.transformed <- preproc.param %>% predict(X_train)
X_train.transformed['Type'] <- Y_train

X_test.transformed <- preproc.param %>% predict(X_test)
X_test.transformed['Type'] <- Y_test

# Applying Linear Discriminant Analysis

library(MASS)

# Fit the model

model <- lda(Type~., data=X_train.transformed)
model

# Make predictions

predictions <- model %>% predict(X_test.transformed)
predictions
str(predictions)
names(predictions)

# Model accuracy

mean(predictions$class==X_test.transformed$Type)
# Accuracy=70.45%

plot(model)

# Create LDA plot using ggplot

lda.data <- cbind(X_train.transformed, predict(model)$x)
ggplot(lda.data, aes(LD1, LD2)) + geom_point(aes(color = Type))

# Quadratic Discriminant Analysis is not used as the sample size is small.
# Applying Multiple Discriminant Analysis

#install.packages("mda")
library(mda)

# Fit the model

modelm <- mda(Type~., data=X_train.transformed)
modelm

# Make predictions

predictionsm <- modelm %>% predict(X_test.transformed)
predictionsm
str(predictionsm)
names(predictionsm)

# Model accuracy

mean(predictionsm==X_test.transformed$Type)
# Accuracy=70.45%

plot(model)

# Applying Flexible Discriminant Analysis

#install.packages("mda")
#library(mda)

# Fit the model

modelf <- fda(Type~., data=X_train.transformed)
modelf

# Make predictions

predictionsf <- modelf %>% predict(X_test.transformed)
predictionsf
str(predictionsf)
names(predictionsf)

# Model accuracy

mean(predictionsf==X_test.transformed$Type)
# Accuracy=70.45%

plot(model)

# Regularized Discriminant Analysis is not applied as the no. of predictors is less than sample size.

