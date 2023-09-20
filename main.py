#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import matplotlib.pyplot as plt
from firebase import firebase
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

# Load the diabetes dataset
diabetes = pd.read_csv('/Users/satvikeltepu/Desktop/diabetes.csv')

def preprocess_data():
    # Split the dataset into training and testing sets
    X_train, X_test, Y_train, Y_test = train_test_split(
        diabetes.loc[:, diabetes.columns != 'Outcome'],
        diabetes['Outcome'],
        stratify=diabetes['Outcome'],
        random_state=66
    )

    # Standardize the dataset
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    return X_train, X_test, Y_train, Y_test

def main():
    # Preprocess the data
    X_train, X_test, Y_train, Y_test = preprocess_data()

    # Initialize and train the Decision Tree Classifier
    tree = DecisionTreeClassifier(criterion='entropy', random_state=0)
    tree.fit(X_train, Y_train)

    try:
        # Request user input for data index
        data_index = int(input("Enter data index: "))

        # Make a prediction
        prediction = tree.predict([X_test[data_index]])

        # Print the prediction
        print(f'Prediction: {prediction[0]}')

        # Initialize Firebase and send the result
        firebase_app = firebase.FirebaseApplication("https://diabetes-857d7.firebaseio.com/", None)
        firebase_data = {'Diabetes': prediction.tolist()}
        result = firebase_app.put("testdata", "LWEis88nRrBK6fJyFHUt", firebase_data)
        print(f'Firebase Result: {result}')
    except ValueError:
        print("Invalid input. Please enter a valid data index.")

if __name__ == '__main__':
    main()
