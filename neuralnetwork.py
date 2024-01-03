import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report

# Load your dataset
# Assuming 'data.csv' is your dataset file
df = pd.read_csv('output_selected_with_categorical_withoutMI.csv')
print("działa")

# Select the relevant columns for training and testing
features = df.drop('FullTimeResult', axis=1)
target = df['FullTimeResult']
#target.to_excel('target.xlsx', index=False)

# Convert categorical variables to numerical using one-hot encoding
features = pd.get_dummies(features)

# Split the data into training and testing sets
X_train = features.iloc[:3800, :]
y_train = target.iloc[:3800]

X_test = features.iloc[-380:, :]
y_test = target.iloc[-380:]

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Build the neural network model
# Create and train the Gaussian Naive Bayes model
model = GaussianNB()
model.fit(X_train_scaled, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test_scaled)

# Print accuracy and classification report
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')
print(classification_report(y_test, y_pred))
