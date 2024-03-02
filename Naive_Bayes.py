# Naive Bayes algorithm:

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

df = pd.read_csv('output_selected_with_categorical_withoutMI.csv')

# Select the relevant columns for training and testing
features = df.drop('FullTimeResult', axis=1)
target = df['FullTimeResult']

# Convert categorical variables to numerical using one-hot encoding
features = pd.get_dummies(features)
# Min Max Scaler
numeric_transformer = MinMaxScaler()
features_transformed = pd.DataFrame(numeric_transformer.fit_transform(features), columns=features.columns)

# Split the data into training and testing sets
X_train = features_transformed.iloc[:3800, :]
y_train = target.iloc[:3800]

X_test = features_transformed.iloc[-380:, :]
y_test = target.iloc[-380:]


var_smooting = np.logspace(0,-9, num=100)

# Create and train the Gaussian Naive Bayes model
model = GaussianNB()
for smooth in var_smooting:
    model.var_smoothing = smooth
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"Accuracy on the test set: {accuracy * 100:.2f}%")
    print(f"var_smoothing: {model.var_smoothing}")
    report = classification_report(y_test, predictions, zero_division=1)
    print("Classification Report:\n", report)
    matrix = confusion_matrix(y_test, predictions)
    print("Confusion matrix:\n", matrix)
    f1score = f1_score(y_test, predictions, average='weighted')
    print(f"f1score on the test set: {f1score * 100:.2f}%")
