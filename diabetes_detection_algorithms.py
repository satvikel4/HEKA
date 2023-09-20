#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# Load the diabetes dataset
diabetes = pd.read_csv('/Users/satvikeltepu/Desktop/diabetes.csv')

def train_and_evaluate_models(X_train, Y_train):
    # Initialize and train machine learning models
    log = LogisticRegression(random_state=0)
    log.fit(X_train, Y_train)

    knn = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
    knn.fit(X_train, Y_train)

    svc_lin = SVC(kernel='linear', random_state=0)
    svc_lin.fit(X_train, Y_train)

    svc_rbf = SVC(kernel='rbf', random_state=0)
    svc_rbf.fit(X_train, Y_train)

    gauss = GaussianNB()
    gauss.fit(X_train, Y_train)

    tree = DecisionTreeClassifier(criterion='entropy', random_state=0)
    tree.fit(X_train, Y_train)

    forest = RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=0)
    forest.fit(X_train, Y_train)

    # Evaluate and print model accuracies
    models = {
        'Logistic Regression': log,
        'K Nearest Neighbor': knn,
        'SVM (Linear Classifier)': svc_lin,
        'SVM (RBF Classifier)': svc_rbf,
        'Gaussian Naive Bayes': gauss,
        'Decision Tree Classifier': tree,
        'Random Forest Classifier': forest
    }

    for model_name, model in models.items():
        accuracy = model.score(X_test, Y_test)
        print(f'{model_name} Test Accuracy: {accuracy:.2f}')

if __name__ == '__main__':
    # Split the dataset into training and testing sets
    X_train, X_test, Y_train, Y_test = train_test_split(
        diabetes.loc[:, diabetes.columns != 'Outcome'],
        diabetes['Outcome'],
        stratify=diabetes['Outcome'],
        random_state=66
    )

    # Train and evaluate machine learning models
    train_and_evaluate_models(X_train, Y_train)
