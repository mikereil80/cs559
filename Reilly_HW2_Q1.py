# Michael Reilly
# I pledge my honor that I have abided by the Stevens Honor System.

import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
import math

# Command line usage: python3 Reilly_HW2_Q1.py

# A classifier using the Maximum Likelihood Estimator is the bayes classifier
# In this case we'll use the importable Gaussian Naive Bayes classifier
# Got the idea to use this from https://towardsdatascience.com/bayes-classifier-with-maximum-likelihood-estimation-4b754b641488
# Imported the scikit_learn Gaussian Naive Bayes Classifier in python

def mean(list):
    added=0
    i=0
    while i<10:
        added+=list[i]
        i+=1
    return added/10

def stdev(list):
    standard_deviation=0
    i=0
    while i<10:
        standard_deviation+=(list[i]-mean(list))**2
        i+=1
    std_dev=standard_deviation/9
    std_deviation=math.sqrt(std_dev)
    return std_deviation

csv=pd.read_csv("pima-indians-diabetes.csv")
columns=csv.iloc[:,1:4]
classifier=csv.iloc[:,-1]
# Documentation for GaussianNB(): https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html
MLE=GaussianNB()
accuracy=[]
i=1

while i<=10:
    x_train,x_test,y_train,y_test=train_test_split(columns,classifier,test_size=0.5)
    MLE.fit(x_train,y_train)
    accuracy.append(MLE.score(x_test,y_test))
    i+=1

print("Mean: ",mean(accuracy))
print("Standard Deviation: ",stdev(accuracy))