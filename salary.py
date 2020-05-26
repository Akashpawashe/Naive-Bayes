import pandas as pd
import numpy as np

from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix



################## Reading the Salary Data 
salary_train = pd.read_csv("D:\\excelR\\Data science notes\\Naive bayes\\asgmnt\\SalaryData_Train.csv")
salary_train.columns
salary_train.shape
salary_train.head()
salary_train.info()
salary_train.describe()
salary_test = pd.read_csv("D:\\excelR\\Data science notes\\Naive bayes\\asgmnt\\SalaryData_Test.csv")
salary_test.columns
salary_test.shape
salary_test.head()
salary_test.info()
salary_test.describe()


string_columns=["workclass","education","maritalstatus","occupation","relationship","race","sex","native"]

from sklearn import preprocessing
number = preprocessing.LabelEncoder()
for i in string_columns:
    salary_train[i] = number.fit_transform(salary_train[i])
    salary_test[i] = number.fit_transform(salary_test[i])

colnames = salary_train.columns
len(colnames[0:13])

trainX = salary_train[colnames[0:13]]
trainY = salary_train[colnames[13]]
testX  = salary_test[colnames[0:13]]
testY  = salary_test[colnames[13]]

from sklearn.naive_bayes import MultinomialNB as MB
from sklearn.naive_bayes import GaussianNB as GB

sgnb = GaussianNB()
smnb = MultinomialNB()
spred_gnb = sgnb.fit(trainX,trainY).predict(testX)
confusion_matrix(testY,spred_gnb)
print ("Accuracy",(10759+1209)/(10759+601+2491+1209)) # 80%

spred_mnb = smnb.fit(trainX,trainY).predict(testX)
confusion_matrix(testY,spred_mnb)
print("Accuracy",(10891+780)/(10891+780+2920+780))  # 75%


# Stratified Method
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
metric_names = ['f1', 'roc_auc', 'average_precision', 'accuracy', 'precision', 'recall']
scores_df = pd.DataFrame(index=metric_names, columns=['Random-CV', 'Stratified-CV']) # to store the Scores
cv = KFold(n_splits=2)
scv = StratifiedKFold(n_splits=2)

trainY = trainY .astype('category')
trainY  = trainY .cat.codes

#Y_train['Salary'] = Y_train['Salary'].astype('category')
#Y_train["Salary"] = Y_train["Salary"].cat.codes

# Multinomial Navie Bayes classifier
for metric in metric_names:
    score1 = cross_val_score(smnb, trainX, trainY, scoring=metric, cv=cv).mean()
    score2 = cross_val_score(smnb, trainX, trainY, scoring=metric, cv=scv).mean()
    scores_df.loc[metric] = [score1, score2]
print(scores_df)

# Gaussian Navie Bayes classifier
for metric in metric_names:
    score1 = cross_val_score(sgnb, trainX, trainY, scoring=metric, cv=cv).mean()
    score2 = cross_val_score(sgnb, trainX,trainY, scoring=metric, cv=scv).mean()
    scores_df.loc[metric] = [score1, score2]
print(scores_df)



