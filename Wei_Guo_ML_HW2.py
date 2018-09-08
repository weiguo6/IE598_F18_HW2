#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep   15:40:16 2018

@author: guowei
"""


#import all package used by this program
import pandas as pd
import os
import numpy as np
from sklearn import datasets 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier 
from sklearn.metrics import accuracy_score
from matplotlib.colors import ListedColormap 
import matplotlib.pyplot as plt
from pydotplus import graph_from_dot_data
from sklearn.tree import export_graphviz
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics


#change the working path to Desktop
#path = '/Users/guowei/Desktop'  #this may need to change on different computers
#os.chdir(path)

#get the features and label data and print number of unique labels
iris = datasets.load_iris()
X = iris.data[:, [2, 3]]
y = iris.target
print('Class labels:', np.unique(y)) 

#split the data into train and test set and print numbers in each set
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.3, 
                                                    random_state=1, 
                                                    stratify=y)
print('Labels counts in y:', np.bincount(y))
print('Labels counts in y_train:', np.bincount(y_train))
print('Labels counts in y_test:', np.bincount(y_test))

#standardize the features
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

#define a function to plot the decision regions
def plot_decision_regions(X, y, classifier, test_idx = None, resolution = 0.02):

    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan') 
    cmap = ListedColormap(colors[:len(np.unique(y))])
    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
    np.arange(x2_min, x2_max, resolution)) 
    Z = classifier.predict(np.array([xx1.ravel(),xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap) 
    plt.xlim(xx1.min(), xx1.max()) 
    plt.ylim(xx2.min(), xx2.max())
    for idx, cl in enumerate(np.unique(y)): 
        plt.scatter(x=X[y == cl, 0], 
                    y=X[y == cl, 1],
                    alpha=0.8, 
                    c=colors[idx], 
                    marker=markers[idx], 
                    label=cl, 
                    edgecolor='black')

    X_test, y_test = X[test_idx, :], y[test_idx]
    plt.scatter(X_test[:, 0], X_test[:, 1],
    c='', edgecolor='black', alpha=1.0,
    linewidth=1, marker='o', s=100, label='test set')

#build a DT model
tree = DecisionTreeClassifier(criterion='gini',max_depth=4, random_state=1)
tree.fit(X_train, y_train)
X_combined = np.vstack((X_train, X_test))
y_combined = np.hstack((y_train, y_test))

#plot the decision regions
plot_decision_regions(X_combined, y_combined, classifier=tree, test_idx=range(105, 150))
plt.xlabel('petal length [cm]') 
plt.ylabel('petal width [cm]') 
plt.legend(loc='upper left') 
plt.show()

#draw the image of DT model
dot_data = export_graphviz(tree,
                           filled = True,
                           rounded = True,
                           class_names = ['Setosa','Versicolor','Virginica'],
                           feature_names = ['petal length','petal width'],
                           out_file = None)
graph = graph_from_dot_data(dot_data)

#save the image to Desktop
graph.write_png('tree.png')

#build a KNN model 
knn = KNeighborsClassifier(n_neighbors = 5, p = 2, metric = 'minkowski')
knn.fit(X_train_std, y_train)
X_combined_std = np.vstack((X_train_std, X_test_std))
plot_decision_regions(X_combined_std, 
                      y_combined, 
                      classifier = knn, 
                      test_idx = range(105,150))

plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc='upper left')
plt.show()

#print the accuracy_score, classification_report,confusion matrix
y_pred = knn.predict(X_test_std)

print("the accruacy score of this KNN model is: ")
print(metrics.accuracy_score(y_test,y_pred))
print("the classification report: ")
print(metrics.classification_report(y_test,y_pred,target_names = iris.target_names))
print("the confusion matrix :")
print(metrics.confusion_matrix(y_test,y_pred))

#try k from 1 - 10 and record the accuracy
#k_range = range(1,26)
scores = []
k_record = []
for k in range(1,10):
    knn = KNeighborsClassifier(n_neighbors = k,p = 2, metric = 'minkowski')
    knn.fit(X_train_std, y_train)
    y_pred = knn.predict(X_test_std)
    scores.append(metrics.accuracy_score(y_test,y_pred))
    k_record.append(k)
    
k_score = pd.DataFrame({'K':k_record,'Score':scores})
print(k_score)

#print Name and ID
print("My name is Wei Guo")
print("My NetID is: weiguo6 ")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")