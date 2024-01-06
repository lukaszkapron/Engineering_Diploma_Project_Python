import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, multilabel_confusion_matrix, precision_score,recall_score, f1_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer

# Load your dataset
# Assuming 'data.csv' is your dataset file
df = pd.read_csv('output_selected_with_categorical_withoutMI.csv')

# Select the relevant columns for training and testing
features = df.drop('FullTimeResult', axis=1)
target = df['FullTimeResult']
#target.to_excel('target.xlsx', index=False)

# Convert categorical variables to numerical using one-hot encoding
features = pd.get_dummies(features)

# Create transformers for numerical and categorical columns
numeric_transformer = MinMaxScaler()

features_transformed = pd.DataFrame(numeric_transformer.fit_transform(features), columns=features.columns)



# Split the data into training and testing sets
X_train = features_transformed.iloc[:3800, :]
y_train = target.iloc[:3800]

X_test = features_transformed.iloc[-380:, :]
y_test = target.iloc[-380:]

# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)

# Standardize the features
# Apply one-hot encoding and standardization
# X_train_transformed = preprocessor.fit_transform(X_train)
# X_test_transformed = preprocessor.transform(X_test)

# pd.DataFrame(X_train).to_excel('X_train_transformed.xlsx', index=False)
# pd.DataFrame(X_test).to_excel('X_test_transformed.xlsx', index=False)

kernel = ['rbf']
gamma = [0.1]

c = [0.1]
degree = [2]
decision_function_shape = ['ovo', 'ovr']

svm_model = SVC(random_state=42, class_weight='balanced')
for x in c:
    for shape in decision_function_shape:
        for kern in kernel:
            for gam in gamma:
                for deg in degree:
                    svm_model.degree = deg
                    svm_model.gamma = gam
                    svm_model.kernel = kern
                    svm_model.decision_function_shape = shape
                    svm_model.c = x
                    svm_model.fit(X_train, y_train)
                    predictions = svm_model.predict(X_test)
                    accuracy = accuracy_score(y_test, predictions)
                    print(f"Accuracy on the test set: {accuracy * 100:.2f}%")
                    print(f"c: {svm_model.c}, decision_function_shape: {svm_model.decision_function_shape}, kernel: {svm_model.kernel}, gamma {svm_model.gamma}, degree: {svm_model.degree}")
                    matrix = confusion_matrix(y_test, predictions)
                    print("Confusion matrix:\n", matrix)
                    matrix = multilabel_confusion_matrix(y_test, predictions)
                    print("multilabel_confusion_matrix matrix:\n", matrix)
                    f1score = f1_score(y_test, predictions, average='weighted')
                    print(f"f1score on the test set: {f1score * 100:.2f}%")
                    report = classification_report(y_test, predictions, zero_division=1)
                    print("Classification Report:\n", report)
from sklearn.model_selection import cross_val_score
# best_model = SVC(random_state=42, kernel='rbf', C=0.1, degree=2, decision_function_shape='ovo', gamma=0.1, class_weight='balanced')
# # cross_val_scores = cross_val_score(best_model, X_train, y_train, cv=10)  # Adjust the number of folds as needed
# # # Print cross-validation scores
# # print("Cross-Validation Scores:", cross_val_scores)
# # print("Mean Cross-Validation Score:", cross_val_scores.mean())
#
# best_model.fit(X_train, y_train)
# predictions = best_model.predict(X_test)
# accuracy = accuracy_score(y_test, predictions)
# print(f"Accuracy on the test set: {accuracy * 100:.2f}%")
# report = classification_report(y_test, predictions, zero_division=1)
# print("Classification Report:\n", report)
# matrix = confusion_matrix(y_test, predictions)
# print("Confusion matrix:\n", matrix)
# matrix = multilabel_confusion_matrix(y_test, predictions)
# print("multilabel_confusion_matrix matrix:\n", matrix)
# f1score = f1_score(y_test, predictions, average='weighted')
# print(f"f1score on the test set: {f1score * 100:.2f}%")







#Hyperparameter tuning using Grid Search
# svm_model.fit(X_train, y_train)
#
# # Get the best hyperparameters
# # best_params = svm_model.best_params_
# # print(f'Best Hyperparameters: {best_params}')
#
# # Get the best SVM model
# # best_svm_model = svm_model.best_estimator_
#
# # Make predictions on the test set using the best model
# y_pred = svm_model.predict(X_test_transformed)
#
# # Print accuracy and classification report
# accuracy = accuracy_score(y_test, y_pred)
# print(f'Accuracy: {accuracy:.2f}')
# print(classification_report(y_test, y_pred))
