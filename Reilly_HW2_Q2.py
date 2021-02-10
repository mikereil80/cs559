# Michael Reilly
# I pledge my honor that I have abided by the Stevens Honor System.

import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import math

# Command line usage: python3 Reilly_HW2_Q2.py

# Imported the scikit_learn KNeighborsClassifier for KNN classifiers in python

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
accuracy1=[]
accuracy5=[]
accuracy11=[]
i=1

while i<=10:
    x_train,x_test,y_train,y_test=train_test_split(columns,classifier,test_size=0.5)
    # Documentation for KNeighborsClassifier(): https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
    knn1=KNeighborsClassifier(n_neighbors=1)
    knn5=KNeighborsClassifier(n_neighbors=5)
    knn11=KNeighborsClassifier(n_neighbors=11)
    knn1.fit(x_train,y_train)
    knn5.fit(x_train,y_train)
    knn11.fit(x_train,y_train)
    accuracy1.append(knn1.score(x_test,y_test))
    accuracy5.append(knn5.score(x_test,y_test))
    accuracy11.append(knn11.score(x_test,y_test))
    i+=1

print("For k=1")
print("Mean: ",mean(accuracy1))
print("Standard Deviation: ",stdev(accuracy1))
print("\n")
print("For k=5")
print("Mean: ",mean(accuracy5))
print("Standard Deviation: ",stdev(accuracy5))
print("\n")
print("For k=11")
print("Mean: ",mean(accuracy11))
print("Standard Deviation: ",stdev(accuracy11))