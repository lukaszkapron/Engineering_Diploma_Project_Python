import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Load your dataset
# Assuming 'data.csv' is your dataset file
df = pd.read_csv('output_selected_with_categorical_withoutMI.csv')
print("działa")

# Select the relevant columns for training and testing
features = df.drop('FullTimeResult', axis=1)
target = df['FullTimeResult']
#target.to_excel('target.xlsx', index=False)

# Convert categorical variables to numerical using one-hot encoding
#features = pd.get_dummies(features)
# Identify categorical columns
categorical_cols = features.select_dtypes(include=['object']).columns

# Create transformers for numerical and categorical columns
numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

# Create a column transformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, features.select_dtypes(include=['float64', 'int64']).columns),
        ('cat', categorical_transformer, categorical_cols)
    ])


# Split the data into training and testing sets
X_train = features.iloc[:3800, :]
y_train = target.iloc[:3800]

X_test = features.iloc[-380:, :]
y_test = target.iloc[-380:]

# Standardize the features
# Apply one-hot encoding and standardization
X_train_transformed = preprocessor.fit_transform(X_train)
X_test_transformed = preprocessor.transform(X_test)



# Create a parameter grid for GridSearchCV
param_grid = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                     'C': [1, 10, 100, 1000]},
                    {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]
# Create and train the SVM model with GridSearchCV
# svm_model = GridSearchCV(SVC(), param_grid, cv=5)

svm_model = SVC()
svm_model.fit(X_train_transformed, y_train)

# Get the best hyperparameters
# best_params = svm_model.best_params_
# print(f'Best Hyperparameters: {best_params}')

# Get the best SVM model
# best_svm_model = svm_model.best_estimator_

# Make predictions on the test set using the best model
y_pred = svm_model.predict(X_test_transformed)

# Print accuracy and classification report
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')
print(classification_report(y_test, y_pred))
