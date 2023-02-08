#DSE INDIVIDUAL ASSIGNMENT
#SHE CHEE YEE A0240383L

#clear workspace
rm(list = ls())

#libraries
library(readr)
library(pls) 
library(corrplot)
library(ROCR)
library(logisticPCA)
library(tree)
library(rpart)
library(rpart.plot)
library(caret)
library(e1071)
library(kknn)



#import data

bank_additional <- read_delim("bank-additional.csv", 
                              delim = ";", escape_double = FALSE, trim_ws = TRUE)
View(bank_additional)
summary(bank_additional)
df = bank_additional
attach(df)


#checking for na values
sum(is.na(df))
#checking for duplicated data
sum(duplicated(df))

#number of yes and no
barplot(table(y), col = c("red","green"))
sum(y == "no")/length(y) #89% of no.

#plotting barplot for various categorical variables
par(mfrow = c(3:2))
barplot(table(df[,c("y","marital")]),main = "Subscription by Marital status",xlab = "Class",col = c("red","green"), beside = TRUE)
legend("topright",c("Not subscribed","Subscribed"),fill = c("red","green"))
barplot(table(df[,c("y","job")]),main = "Subscription by Job",xlab = "Class",col = c("red","green"), beside = TRUE)
barplot(table(df[,c("y","education")]),main = "Subscription by Education",xlab = "Class",col = c("red","green"), beside = TRUE)
barplot(table(df[,c("y","default")]),main = "Subscription by Default Status",xlab = "Class",col = c("red","green"), beside = TRUE)
barplot(table(df[,c("y","housing")]),main = "Subscription by Housing Status",xlab = "Class",col = c("red","green"), beside = TRUE)
barplot(table(df[,c("y","loan")]),main = "Subscription by Loan status",xlab = "Class",col = c("red","green"), beside = TRUE)

par(mfrow = c(3,2))
barplot(table(df[,c("y","contact")]),main = "Subscription by Contact",xlab = "Class",col = c("red","green"), beside = TRUE)
legend("topright",c("Not subscribed","Subscribed"),fill = c("red","green"))
barplot(table(df[,c("y","month")]),main = "Subscription by Month",xlab = "Class",col = c("red","green"), beside = TRUE)
barplot(table(df[,c("y","day_of_week")]),main = "Subscription by Day",xlab = "Class",col = c("red","green"), beside = TRUE)
barplot(table(df[,c("y","poutcome")]),main = "Subscription by previous outcome",xlab = "Class",col = c("red","green"), beside = TRUE)

#from common knowledge and the barplot, it seems like the day of the week will not affect our resutls much, hence we are removing it
df = subset(df, select = -c(day_of_week))
#this attribute highly affects the output target (e.g., if duration=0 then y="no"). Yet, the duration is not known before hand.
df = subset(df, select = -c(duration))

#removing some unwanted data and outlier
outlier_row = which(education == "illiterate")
outlier_row
df = df[-outlier_row, ]

outlier_row = which(default == "yes")
outlier_row
df = df[-outlier_row, ]


# I will be exploring this data set to determine what are the factors affecting whether
# a customer's buying decision for financial products.


bank_additional_numeric = subset(df, select = -c(job,marital, education, default,
                                                 housing,loan,contact,month,
                                                 poutcome,y))
summary(df)
#based on the summary, there are a total of 9 categorical data and 9 numerical data

#check distribution of numerical predictors
par(mfrow = c(3:2))
plot(density(age), frame = FALSE, col= "steelblue", main = "Density plot of age")
plot(density(campaign), frame = FALSE, col= "steelblue", main = "Density plot of campaign")
plot(density(pdays), frame = FALSE, col= "steelblue", main = "Density plot of pdays")
plot(density(previous), frame = FALSE, col= "steelblue", main = "Density plot of previous")
plot(density(emp.var.rate), frame = FALSE, col= "steelblue", main = "Density plot of emp.var.rate")
plot(density(cons.price.idx), frame = FALSE, col= "steelblue", main = "Density plot of cons.price.idx")
par(mfrow = c(2,2))
plot(density(cons.conf.idx), frame = FALSE, col= "steelblue", main = "Density plot of cons.conf.idx")
plot(density(euribor3m), frame = FALSE, col= "steelblue", main = "Density plot of euribor3m")
plot(density(nr.employed), frame = FALSE, col= "steelblue", main = "Density plot of nr.employed")

# Visualize the pair-wise correlation matrix
par(mfrow = c(1,1))
cor_matrix = round(cor(bank_additional_numeric), 2)
corrplot(cor_matrix, type = "upper", order = 'alphabet',
         tl.srt = 45, tl.cex = 0.9,   
         method = "circle")


#PCA
df1 = df
df1$y = ifelse(df1$y == "yes", 1, 0)
df1 = subset(df1, select = -c(job,marital, education, default,
                             housing,loan,contact,month,
                             poutcome))
prall = prcomp(df1, scale = TRUE)

biplot(prall, main = "Biplot") #biplot

prall.s = summary(prall)
prall.s$importance
scree = prall.s$importance[2,]

plot(scree, main = "Scree Plot", xlab = "Principal Component", 
     ylab = "Proportion of Variance Explained", ylim = c(0,1), type = 'b', cex = .8)
#using PCA to reduce dimensionality did not go well here as we are not able to exclude alot of components

#prediction with PCR
ntrain = 3200
set.seed(103)
tr = sample(1:nrow(df),ntrain)  # draw ntrain observations from original data

train1 = df1[tr,]   # Training sample
test1 = df1[-tr,]

pcr.fit=pcr(y~.,data=train1, scale=TRUE, validation="CV")


plot(pcr.fit, "loadings", comps = 1:9, legendpos = "topleft")
abline(h = 0) #add the zero line for reference

validationplot(pcr.fit, val.type="MSEP", main="CV",legendpos = "topright")

pcr.pred=predict(pcr.fit, newdata=test1, ncomp=7)
mean((test1$y-pcr.pred)^2) #MSE for PCR
table(pcr.pred > 0.5, test1$y)#confusion matrix

pred_pcr = prediction(pcr.pred[1:917], test1$y)
perf_pcr = performance(pred_pcr, measure = "tpr", x.measure = "fpr")
auc_perf_pcr = performance(pred_pcr, measure = "auc") # Calculate AUC
plot(perf_pcr, col = "steelblue", lwd = 2, main="ROC for PCR") # Plot ROC curve
abline(0, 1, lwd = 1, lty = 2) # Add dashed diagonal line
text(0.4, 0.85, paste("AUC =", round(auc_perf_pcr@y.values[[1]], 2)))


###logistic pca (only numerialc data)


logpca_cv = cv.lpca(train1, ks = 2, ms = 1:10)
plot(logpca_cv) #CV PLOT
logpca_fit = logisticPCA(train1, k = 2, m = which.min(logpca_cv))

logpca_pred = predict(logpca_fit, test1, type = "response")
table(logpca_pred[,10]>0.5,test1$y)#confusion matrix

predLGPCA = prediction(logpca_pred[,10], test1$y)
perfLGPCA = performance(predLGPCA, measure = "tpr", x.measure = "fpr")
auc_perfLGPCA = performance(predLGPCA, measure = "auc") # Calculate AUC
plot(perfLGPCA, col = "steelblue", lwd = 2, main="ROC for LogPca") # Plot ROC curve
abline(0, 1, lwd = 1, lty = 2) # Add dashed diagonal line
text(0.4, 0.8, paste("AUC =", round(auc_perfLGPCA@y.values[[1]], 2)))




#decision trees 

ntrain = 3200
set.seed(103)
tr = sample(1:nrow(df),ntrain)  # draw ntrain observations from original data
train = df[tr,]   # Training sample
test = df[-tr,]
treeGini= rpart(y~.,data=train, method = "class", minsplit = 10, cp = .0005, maxdepth = 30)


plotcp(treeGini)
bestcp=treeGini$cptable[which.min(treeGini$cptable[,"xerror"]),"CP"]
bestGini = prune(treeGini,cp=bestcp)
rpart.plot(bestGini, shadow.col = "gray")
text(bestGini,digits=4,use.n=TRUE,fancy=FALSE,bg='lightblue')
treepred = predict(bestGini,newdata = test)

table(treepred[,2]>0.5,test$y)#confusion matrix


pred = prediction(treepred[,2], test$y)
perfDT = performance(pred, measure = "tpr", x.measure = "fpr")
auc_perfDT = performance(pred, measure = "auc") # Calculate AUC
plot(perfDT, col = "steelblue", lwd = 2, main="ROC for Tree") # Plot ROC curve
abline(0, 1, lwd = 1, lty = 2) # Add dashed diagonal line
text(0.4, 0.8, paste("AUC =", round(auc_perfDT@y.values[[1]], 2)))




#from the decision tree we found out that a few predictors are more prominent than the others, namely 
#number of employee in the company , pdays 




####
#NAIVE BAYES
ntrain = 3200
set.seed(103)
tr = sample(1:nrow(df),ntrain)  # draw ntrain observations from original data
train = df[tr,]   # Training sample
test = df[-tr,]

nbfit = naiveBayes(y~., data=train)
nbpred=predict(nbfit, test, type="class")
nbpred2=predict(nbfit, test, type="raw")

table(nbpred, test$y)#confusion matrix

pred = prediction(nbpred2[,2], test$y) #second coloumn corresponds to 'yes', Y=1
perfNB = performance(pred, measure = "tpr", x.measure = "fpr")
auc_perfNB = performance(pred, measure = "auc") # Calculate AUC
plot(perfNB, col = "steelblue", lwd = 2, main = "ROC for NB") # Plot ROC curve
abline(0, 1, lwd = 1, lty = 2) # Add dashed diagonal line
text(0.4, 0.9, paste("AUC =", round(auc_perfNB@y.values[[1]], 2))) # Compute AUC and add text to ROC plot.


############
#first try to use logistic regression with all variables
ntrain = 3200
set.seed(103)
tr = sample(1:nrow(df),ntrain)  # draw ntrain observations from original data
train = df[tr,]   # Training sample
test = df[-tr,]
df$y = ifelse(df$y == "yes", 1, 0)
ntrain = 3200
set.seed(103)
tr = sample(1:nrow(df),ntrain)  # draw ntrain observations from original data
train = df[tr,]   # Training sample
test = df[-tr,]


logitfit = glm(y~.,data = train, family = binomial)
summary(logitfit)
logitpred = predict(logitfit, newdata = test, type = "response")
pred_lg1 = prediction(logitpred, test$y)
perf_lg1 = performance(pred_lg1, measure = "tpr", x.measure = "fpr")
auc_perf_lg1 = performance(pred_lg1, measure = "auc") # Calculate AUC
plot(perf_lg1, col = "steelblue", lwd = 2, main="ROC for Logit") # Plot ROC curve
abline(0, 1, lwd = 1, lty = 2) # Add dashed diagonal line
text(0.4, 0.85, paste("AUC =", round(auc_perf_lg1@y.values[[1]], 2)))

table(logitpred > 0.5, test$y)#confusion matrix


#fitting logit w the predictors the decision tree have picked out
logitfit2 = glm(y~ nr.employed+pdays,
                data = train, family = binomial)
summary(logitfit2)
logitpred2 = predict(logitfit2, newdata = test, type = "response")
pred_lg2 = prediction(logitpred2, test$y)
perf_lg2 = performance(pred_lg2, measure = "tpr", x.measure = "fpr")
auc_perf_lg2 = performance(pred_lg2, measure = "auc") # Calculate AUC
plot(perf_lg2, col = "steelblue", lwd = 2, main="ROC for Logit2") # Plot ROC curve
abline(0, 1, lwd = 1, lty = 2) # Add dashed diagonal line
text(0.4, 0.85, paste("AUC =", round(auc_perf_lg2@y.values[[1]], 2)))
table(logitpred2 > 0.5, test$y) #confusion matrix

#here we can see that the performance is not as good as when using all variables.
#through domain knowledge we know that job can directly/indirectly affect ones likeliness to buy a financial product
#hence we are going to add in job as a predictor.

logitfit3 = glm(y~ nr.employed+pdays+job,
                data = train, family = binomial)
summary(logitfit3)
logitpred3 = predict(logitfit3, newdata = test, type = "response")
pred_lg3 = prediction(logitpred3, test$y)
perf_lg3 = performance(pred_lg3, measure = "tpr", x.measure = "fpr")
auc_perf_lg3 = performance(pred_lg3, measure = "auc") # Calculate AUC
plot(perf_lg3, col = "steelblue", lwd = 2, main="ROC for Logit3") # Plot ROC curve
abline(0, 1, lwd = 1, lty = 2) # Add dashed diagonal line
text(0.4, 0.85, paste("AUC =", round(auc_perf_lg3@y.values[[1]], 2)))



#knn (only numerical data)
ntrain = 3200
set.seed(103)
tr = sample(1:nrow(df),ntrain)  # draw ntrain observations from original data
train1 = df1[tr,]   # Training sample
test1 = df1[-tr,]


knnpred  =  kknn(y~.,train1,test1,k=20,kernel = "rectangular")

table(knnpred$fitted.values > 0.5, test1$y) #confusion matrix
pred = prediction(knnpred$fitted.values, test1$y)
perfKNN = performance(pred, measure = "tpr", x.measure = "fpr")
auc_perfKNN = performance(pred, measure = "auc") # Calculate AUC
plot(perfKNN, col = "steelblue", lwd = 2, main="ROC for KNN") # Plot ROC curve
abline(0, 1, lwd = 1, lty = 2) # Add dashed diagonal line
text(0.4, 0.9, paste("AUC =", round(auc_perfKNN@y.values[[1]], 2))) #AUC = 0.76




################
#ALL ROC PLOTS
################
plot(perfDT, col = "blue", lwd = 2, main = "All ROC")
text(0.1, 1, paste("DT AUC =", round(auc_perfDT@y.values[[1]], 2)), col = "blue")
plot( perfKNN, add = TRUE, col = "red", lwd = 2)
text(0.1, 0.95, paste("KNN AUC =", round(auc_perfKNN@y.values[[1]], 2)), col = "red")
plot( perfNB, add = TRUE, col = "seagreen", lwd = 2)
text(0.1, 0.9, paste("NB AUC =", round(auc_perfNB@y.values[[1]], 2)), col = "seagreen")
plot( perf_lg1, add = TRUE, col = "mediumpurple3", lwd = 2)
text(0.1, 0.85, paste("LOGIT AUC =", round(auc_perf_lg3@y.values[[1]], 2)), col = "mediumpurple3")
plot( perf_pcr, add = TRUE, col = "black", lwd = 2)
text(0.1, 0.80, paste("PCR AUC =", round(auc_perf_pcr@y.values[[1]], 2)), col = "black")
plot( perfLGPCA, add = TRUE, col = "khaki4", lwd = 2)
text(0.1, 0.75, paste("LOGPCA AUC =", round(auc_perfLGPCA@y.values[[1]], 2)), col = "khaki4")
abline(0, 1, lwd = 1, lty = 2) 
legend("bottomright",c("DT","KNN","NB","LOGIT","PCR","LOGPCA"),fill = c("blue","red","seagreen","mediumpurple3","black","khaki4"))








