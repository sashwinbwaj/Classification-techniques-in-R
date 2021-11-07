
# LOGISTIC/KNN/SVM CLASSIFICAION 

library(caTools)
library(ggplot2)
library(ggpubr)
library(ElemStatLearn)

dataset=read.csv("Social_Network_Ads.csv")
dataset=dataset[,3:5]
print(head(dataset,10))

set.seed(123)
training_set=subset(dataset,sample.split(dataset$Purchased,SplitRatio = 0.75)==TRUE)
test_set=subset(dataset,sample.split(dataset$Purchased,SplitRatio = 0.75)==FALSE)
training_set[,1:2]=scale(training_set[,1:2])
test_set[,1:2]=scale(test_set[,1:2])


#Logistic Regression 

logreg=glm(formula = Purchased~Age+EstimatedSalary,data=training_set,family='binomial')
summary(logreg)

training_set$log_prob=predict(logreg,type='response',newdata = training_set[,c('Age', 'EstimatedSalary')])
training_set$log_pred=as.numeric(ifelse(training_set$log_prob>0.5,1,0))

test_set$log_prob=predict(logreg,type='response',newdata = test_set[,c('Age', 'EstimatedSalary')])
test_set$log_pred=as.numeric(ifelse(test_set$log_prob>0.5,1,0))


#KNN CLASSIFICATION
#REMEMEBER THAT FOR KNN, TRAINING SET SHOULD BE PRESENT EVERYTIME WE RUN IT ON TEST

library(class)
test_set$knn_pred=knn(train=training_set[,c('Age','EstimatedSalary')],test=test_set[c('Age','EstimatedSalary')]
           ,cl=training_set[,3],k=5,prob=TRUE)

#Confusion Matrix

cm_train=table(training_set$Purchased,training_set$log_pred)
cm_test=table(test_set$Purchased,test_set$log_pred)

set=test_set
X1=seq(min(set$Age)-1,max(set$Age)+1,by=0.01)
X2=seq(min(set$EstimatedSalary)-1,max(set$EstimatedSalary)+1,by=0.01)
grid_set=expand.grid(X1,X2)
colnames(grid_set)=c('Age','EstimatedSalary')

grid_set$log_prob=predict(logreg,type='response',newdata=grid_set[,c('Age','EstimatedSalary')])
grid_set$log_pred=as.numeric(ifelse(grid_set$log_prob>0.5,1,0))
grid_set$knn_pred=knn(train=training_set[,1:2],test=grid_set[,1:2],cl=training_set[,3],k=5)


#Plotting Logistic Regression

plot(set[,c('Age','EstimatedSalary')],
     main='Logistic Regression (Test Set)',
     xlab='Age',ylab='Estimated Salary',
     xlim=range(X1),ylim=range(X2)) +
contour(X1,X2,matrix(as.numeric(grid_set$log_pred),length(X1),length(X2)),add=TRUE) + 
  points(grid_set, pch= '.', col=ifelse(grid_set$log_pred==1,'springgreen3','tomato')) +
points(set, pch=21, bg=ifelse(set$Purchased==1,'green4','red'))

#Plotting KNN Classification

plot(set[,c('Age','EstimatedSalary')],
     main='KNN Classification (Test Set)',
     xlab='Age',ylab='Estimated Salary',
     xlim=range(X1),ylim=range(X2)) +
  contour(X1,X2,matrix(as.numeric(grid_set$knn_pred),length(X1),length(X2)),add=TRUE) + 
  points(grid_set, pch= '.', col=ifelse(grid_set$knn_pred==1,'springgreen3','tomato')) +
  points(set, pch=21, bg=ifelse(set$Purchased==1,'green4','red'))




# SUPPORT VECTOR MACHINE CLASSIFICATION 

library(caTools)
library(ggplot2)
library(ggpubr)
library(ElemStatLearn)

dataset=read.csv("Social_Network_Ads.csv")
dataset=dataset[,3:5]
print(head(dataset,10))

set.seed(123)
training_set=subset(dataset,sample.split(dataset$Purchased,SplitRatio = 0.75)==TRUE)
test_set=subset(dataset,sample.split(dataset$Purchased,SplitRatio = 0.75)==FALSE)
training_set[,1:2]=scale(training_set[,1:2])
test_set[,1:2]=scale(test_set[,1:2])

# SVM MODEL

library(e1071)
svm_reg = svm(formula = Purchased ~ .,data = training_set, type = 'C-classification',kernel = 'linear')
training_set$svm_pred=predict(svm_reg,newdata = training_set[,c('Age', 'EstimatedSalary')])
test_set$svm_pred=predict(svm_reg,newdata = test_set[,c('Age', 'EstimatedSalary')])  

# KERNAL SVM 

kernal_svm=svm(formula=Purchased~Age+EstimatedSalary,data=training_set,type='C-classification',kernal='radial')
training_set$ksvm_pred=predict(kernal_svm,newdata = training_set[,c('Age', 'EstimatedSalary')])
test_set$ksvm_pred=predict(kernal_svm,newdata = test_set[,c('Age', 'EstimatedSalary')])  

set=test_set
X1=seq(min(set$Age)-1,max(set$Age)+1,by=0.01)
X2=seq(min(set$EstimatedSalary)-1,max(set$EstimatedSalary)+1,by=0.01)
grid_set=expand.grid(X1,X2)
colnames(grid_set)=c('Age','EstimatedSalary')

grid_set$svm_pred=predict(svm_reg,newdata = grid_set[,c('Age', 'EstimatedSalary')])
grid_set$ksvm_pred=predict(kernal_svm,newdata = grid_set[,c('Age', 'EstimatedSalary')])

#Plotting SVM Classification

plot(set[,c('Age','EstimatedSalary')],
     main='SVM Classification (Test Set)',
     xlab='Age',ylab='Estimated Salary',
     xlim=range(X1),ylim=range(X2)) +
  contour(X1,X2,matrix(as.numeric(grid_set$svm_pred),length(X1),length(X2)),add=TRUE) + 
  points(grid_set, pch= '.', col=ifelse(grid_set$svm_pred==1,'springgreen3','tomato')) +
  points(set, pch=21, bg=ifelse(set$Purchased==1,'green4','red'))

#Plotting KERNAL SVM Classification

plot(set[,c('Age','EstimatedSalary')],
     main='KERNAL SVM Classification (Test Set)',
     xlab='Age',ylab='Estimated Salary',
     xlim=range(X1),ylim=range(X2)) +
  contour(X1,X2,matrix(as.numeric(grid_set$ksvm_pred),length(X1),length(X2)),add=TRUE) + 
  points(grid_set, pch= '.', col=ifelse(grid_set$ksvm_pred==1,'springgreen3','tomato')) +
  points(set, pch=21, bg=ifelse(set$Purchased==1,'green4','red'))



# NAIVES BAYES CLASSIFICATION

library(caTools)
library(ggplot2)
library(ggpubr)
library(ElemStatLearn)

dataset=read.csv("Social_Network_Ads.csv")
dataset=dataset[,3:5]
dataset$Purchased=factor(dataset$Purchased,levels=c(0 , 1))
print(head(dataset,10))

set.seed(123)
training_set=subset(dataset,sample.split(dataset$Purchased,SplitRatio = 0.75)==TRUE)
test_set=subset(dataset,sample.split(dataset$Purchased,SplitRatio = 0.75)==FALSE)
training_set[,1:2]=scale(training_set[,1:2])
test_set[,1:2]=scale(test_set[,1:2])


# CREATING THE CLASSIFIIER (NAIVE BAYES) 

library(e1071)
naive_bayes=naiveBayes(x=training_set[,1:2],y=training_set[,3])
summary(naive_bayes)
training_set$naive_pred=predict(naive_bayes,newdata = training_set[,c('Age', 'EstimatedSalary')])
test_set$naive_pred=predict(naive_bayes,newdata = test_set[,c('Age', 'EstimatedSalary')])  

set=test_set
X1=seq(min(set$Age)-1,max(set$Age)+1,by=0.01)
X2=seq(min(set$EstimatedSalary)-1,max(set$EstimatedSalary)+1,by=0.01)
grid_set=expand.grid(X1,X2)
colnames(grid_set)=c('Age','EstimatedSalary')

grid_set$naive_pred=predict(naive_bayes,newdata = grid_set[,c('Age', 'EstimatedSalary')])

#Plotting NAIVE BAYES Classification

plot(set[,c('Age','EstimatedSalary')],
     main='Naive Bayes Classification (Test Set)',
     xlab='Age',ylab='Estimated Salary',
     xlim=range(X1),ylim=range(X2)) +
  contour(X1,X2,matrix(as.numeric(grid_set$naive_pred),length(X1),length(X2)),add=TRUE) + 
  points(grid_set, pch= '.', col=ifelse(grid_set$naive_pred==1,'springgreen3','tomato')) +
  points(set, pch=21, bg=ifelse(set$Purchased==1,'green4','red'))

# DECISION TREES 

library(caTools)
library(ggplot2)
library(ggpubr)
library(ElemStatLearn)

dataset=read.csv("Social_Network_Ads.csv")
dataset=dataset[,3:5]
dataset$Purchased=factor(dataset$Purchased,levels=c(0 , 1))
print(head(dataset,10))

set.seed(123)
training_set=subset(dataset,sample.split(dataset$Purchased,SplitRatio = 0.75)==TRUE)
test_set=subset(dataset,sample.split(dataset$Purchased,SplitRatio = 0.75)==FALSE)
training_set[,1:2]=scale(training_set[,1:2])
test_set[,1:2]=scale(test_set[,1:2])


library(rpart)
library(rpart.plot)

dtree=rpart(formula=Purchased~Age+EstimatedSalary,data=training_set,method='class',control=rpart.control(minbucket=10,minsplit=10))
summary(dtree)
rpart.plot(dtree,extra=106)

training_set$dtree_pred=predict(dtree,newdata = training_set[,1:2],type='class')
test_set$dtree_pred=predict(dtree,newdata=test_set[1:2],type='class')

set=test_set
X1=seq(min(set$Age)-1,max(set$Age)+1,by=0.01)
X2=seq(min(set$EstimatedSalary)-1,max(set$EstimatedSalary)+1,by=0.01)
grid_set=expand.grid(X1,X2)
colnames(grid_set)=c('Age','EstimatedSalary')

grid_set$dtree_pred=predict(dtree,newdata = grid_set[,c('Age', 'EstimatedSalary')],type='class')

plot(set[,c('Age','EstimatedSalary')],
     main='Decision Tree Classification (Test Set)',
     xlab='Age',ylab='Estimated Salary',
     xlim=range(X1),ylim=range(X2)) +
  contour(X1,X2,matrix(as.numeric(grid_set$dtree_pred),length(X1),length(X2)),add=TRUE) + 
  points(grid_set, pch= '.', col=ifelse(grid_set$dtree_pred==1,'springgreen3','tomato')) +
  points(set, pch=21, bg=ifelse(set$Purchased==1,'green4','red'))

# RANDOM FOREST CLASSIFICATION

library(randomForest)

ranforest=randomForest(x=training_set[1:2],y=training_set$Purchased,ntree=10)
test_set$rf_pred=predict(ranforest,newdata = test_set[1:2])
grid_set$rf_pred=predict(ranforest,newdata = grid_set[,c('Age', 'EstimatedSalary')])

plot(set[,c('Age','EstimatedSalary')],
     main='Random Forest Classification (Test Set)',
     xlab='Age',ylab='Estimated Salary',
     xlim=range(X1),ylim=range(X2)) +
  contour(X1,X2,matrix(as.numeric(grid_set$rf_pred),length(X1),length(X2)),add=TRUE) + 
  points(grid_set, pch= '.', col=ifelse(grid_set$rf_pred==1,'springgreen3','tomato')) +
  points(set, pch=21, bg=ifelse(set$Purchased==1,'green4','red'))

