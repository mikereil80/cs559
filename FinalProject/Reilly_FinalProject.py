# Michael Reilly
# I pledge my honor that I have abided by the Stevens Honor System.

# Command line usage: python3 Reilly_FinalProject.py

# The goal is to analyze the iris.data dataset from UCI ML repository.
# This will act similarly to recognizing faces instead it is Object recognition.
# The goal is to teach the machine how to recognize the various types of iris flowers, 
# the setosa, the viriginica, and the versicolor.
# We also will determine which of the three given models, 
# KNN, Logitsitic Regression, or SVM will be the best fit classifier for the data.
# At the end it will be decided which of these classifiers worked best in that instance.
# Testing has shown however that theya re all fairly accurate, an thus the answer is not always the same based on the training and test samples it takes.
# This relates to the class due to us learning about these classifiers during lectures.

# The sklearn kit is used for the classifiers as they are well optimized to get these results quickly and effectively.

# START: OWN CODE

import pandas as pd
import numpy as np
import math
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

# Define the mean and standard deviation functions for future use.
def mean(list):
    added=0
    i=0
    while i<100:
        added+=list[i]
        i+=1
    return added/100

def stdev(list):
    standard_deviation=0
    i=0
    while i<100:
        standard_deviation+=(list[i]-mean(list))**2
        i+=1
    std_dev=standard_deviation/99
    std_deviation=math.sqrt(std_dev)
    return std_deviation

# Start with reading the data file and printing out helpful information about the data.
names=['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset=pd.read_csv('iris.data', names=names)
print("Shape of Data: {}".format(dataset.shape))
print("\n")
print("Statistical Summary of Data: \n{}".format(dataset.describe()))
print("\n")
print("Class Distribution of the Data: \n{}".format(dataset.groupby('class').size()))
print("\n")

# Check for Plots folder and it's existence. 
# If it does not, create it. 
# This Plots folder will be used to store the boxplots created later to see them.
Plots=os.getcwd()+"/Plots"
Check_Folder=os.path.isdir(Plots)
if not Check_Folder:
    os.mkdir(Plots)

# Make a boxplot for visual representation of the data, with respect to class for each data-type.
# First sepal-length v. class boxplot
plt.figure(0)
dataset.boxplot(by='class', column=['sepal-length'])
plt.savefig("Plots/BoxSepalLength.png")
plt.show()

# Next sepal-width v. class boxplot
plt.figure(1)
dataset.boxplot(by='class', column=['sepal-width'])
plt.savefig("Plots/BoxSepalWidth.png")
plt.show()

# Now petal-length v. class boxplot
plt.figure(2)
dataset.boxplot(by='class', column=['petal-length'])
plt.savefig("Plots/BoxPetalLength.png")
plt.show()

# Lastly petal-width v. class boxplot
plt.figure(3)
dataset.boxplot(by='class', column=['petal-width'])
plt.savefig("Plots/BoxPetalWidth.png")
plt.show()

# Now build the train and test samples, 
# which in this case the train will get 75% of the data, and the test will be the other 25% of the data.
values=dataset.values
data=values[:,0:4]
target=values[:,-1]
x_train,x_test, y_train,y_test=train_test_split(data, target, test_size=0.25)
print("X_train shape: {}".format(x_train.shape))
print("Y_train shape: {}".format(y_train.shape))
print("X_test shape: {}".format(x_test.shape))
print("Y_test shape: {}".format(y_test.shape))
print("\n")

# First test the KNN classifier and see how accurate it is.
i=1
knn_accuracy=[]
while i<=100:
    knn=KNeighborsClassifier(n_neighbors=5)
    knn.fit(x_train,y_train)
    knn_accuracy.append(knn.score(x_test,y_test))
    i+=1
print("KNN mean accuracy: ",mean(knn_accuracy))
print("KNN standard deviation of accuracy: ",stdev(knn_accuracy))
print("\n")

# Next test Logistic Regression classifier and see how accurate it is.
j=1
logic_accuracy=[]
while j<=100:
    logic=LogisticRegression()
    logic.fit(x_train,y_train)
    logic_accuracy.append(logic.score(x_test,y_test))
    j+=1
print("Logistic Regression mean accuracy: ",mean(logic_accuracy))
print("Logistic Regression standard deviation of accuracy: ",stdev(logic_accuracy))
print("\n")

# Now test Suppport Vector classifier and see how accurate it is.
k=1
svm_accuracy=[]
while k<=100:
    svm=SVC()
    svm.fit(x_train,y_train)
    svm_accuracy.append(svm.score(x_test,y_test))
    k+=1
print("SVM mean accuracy: ",mean(svm_accuracy))
print("SVM standard deviation of accuracy: ",stdev(svm_accuracy))
print("\n")

# Now we must decide which classifier is the mosta accurate for this data set.
knn_mean=mean(knn_accuracy)
logic_mean=mean(logic_accuracy)
svm_mean=mean(svm_accuracy)
if knn_mean>logic_mean and knn_mean>svm_mean:
    print("KNN is the most accurate classifier in this case.")

if logic_mean>knn_mean and logic_mean>svm_mean:
    print("Logistic Regression is the most accurate classifier in this case.")

if svm_mean>knn_mean and svm_mean>logic_mean:
    print("SVM is the most accurate classifier in this case.")

if knn_mean==logic_mean and knn_mean>svm_mean:
    print("KNN and Logistic Regression are the most accurate classifiers in this case.")

if knn_mean>logic_mean and knn_mean==svm_mean:
    print("KNN and SVM are the most accurate classifiers in this case.")

if logic_mean>knn_mean and logic_mean==svm_mean:
    print("Logistic Regression and SVM are the most accurate classifiers in this case.")

if knn_mean==logic_mean and knn_mean==svm_mean:
    print("KNN, Logistic Regression and SVM are all equally accurate classifiers in this case.")
# END: OWN CODE