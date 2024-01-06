import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, multilabel_confusion_matrix, precision_score,recall_score, f1_score

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
numeric_transformer = MinMaxScaler()
features_transformed = pd.DataFrame(numeric_transformer.fit_transform(features), columns=features.columns)

# Split the data into training and testing sets
X_train = features_transformed.iloc[:3800, :]
y_train = target.iloc[:3800]

X_test = features_transformed.iloc[-380:, :]
y_test = target.iloc[-380:]

# Standardize the features
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)





var_smooting = np.logspace(0,-9, num=100)

# Build the neural network model
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
    # matrix = multilabel_confusion_matrix(y_test, predictions)
    # print("multilabel_confusion_matrix matrix:\n", matrix)
    f1score = f1_score(y_test, predictions, average='weighted')
    print(f"f1score on the test set: {f1score * 100:.2f}%")

# for smooth in var_smooting:
# model.var_smoothing = 1
# model.fit(X_train, y_train)
# predictions = model.predict(X_test)
# accuracy = accuracy_score(y_test, predictions)
# print(f"Accuracy on the test set: {accuracy * 100:.2f}%")
# print(f"var_smoothing: {model.var_smoothing}")
# report = classification_report(y_test, predictions, zero_division=1)
# print("Classification Report:\n", report)
# matrix = confusion_matrix(y_test, predictions)
# print("Confusion matrix:\n", matrix)
# matrix = multilabel_confusion_matrix(y_test, predictions)
# print("multilabel_confusion_matrix matrix:\n", matrix)
#
# # Create a DataFrame with details of predicted rows
# predicted_df = X_test.copy()
#
# predicted_df['True_Label'] = y_test
# predicted_df['Predicted_Label'] = predictions
#
# # Save the DataFrame to an XLS file
# predicted_df.to_excel('predictions.xlsx', index=False)
# print("Predictions saved to 'predictions.xlsx'")
#Hyperparameter tuning using Grid Search
# Make predictions on the test set
#y_pred = model.predict(X_test)

# # Print accuracy and classification report
# accuracy = accuracy_score(y_test, y_pred)
# print(f'Accuracy: {accuracy:.2f}')
# print(classification_report(y_test, y_pred))
