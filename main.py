# # preds = rf.predict(test[predictors])
# # error = classification_report(test["FullTimeResult"], preds)
# # print(error)
# #
# # accuracy = accuracy_score(test["FullTimeResult"], preds)
# # print("Accuracy Score:", accuracy)
# #
# # print("Unique labels in training data:", train["FullTimeResult"].unique())
# # print("Unique labels in test data:", test["FullTimeResult"].unique())
# #
# #
# # Generate a classification report with zero_division parameter set to 1
# report = classification_report(y_test, y_pred, zero_division=1)
# print("Classification Report:\n", report)
#
# # Create a DataFrame with details of predicted rows
# predicted_df = test_data.copy()
# predicted_df['True_Label'] = y_test
# predicted_df['Predicted_Label'] = y_pred
#
# # Save the DataFrame to an XLS file
# predicted_df.to_excel('predictions.xlsx', index=False)
# print("Predictions saved to 'predictions.xlsx'")






import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load the CSV file into a DataFrame
df = pd.read_csv('output.csv')

# Select the relevant columns for training and testing
features = df.drop('FullTimeResult', axis=1)
target = df['FullTimeResult']
target.to_excel('target.xlsx', index=False)

# Convert categorical variables to numerical using one-hot encoding
features = pd.get_dummies(features)

# Split the data into training and testing sets
X_train = features.iloc[:3800, :]
y_train = target.iloc[:3800]

X_test = features.iloc[-380:, :]
y_test = target.iloc[-380:]

# Save training data to Excel
X_train.to_excel('X_train.xlsx', index=False)
y_train.to_excel('y_train.xlsx', index=False)

# Save test data to Excel
X_test.to_excel('X_test.xlsx', index=False)
y_test.to_excel('y_test.xlsx', index=False)

# Standardize the features
# scaler = StandardScaler()
# X_train_standardized = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
# X_test_standardized = scaler.transform(X_test)

print("standarized")
# Initialize and train a Random Forest classifier
# best_model = RandomForestClassifier(n_estimators=150, random_state=42, max_depth=10, min_samples_leaf=4, min_samples_split=2)  # You can adjust n_estimators as needed
# best_model.fit(X_train, y_train)

import pandas as pd

# Assuming you have already loaded your data and preprocessed it as in your code

# Concatenate the features and target into a single DataFrame
all_data = pd.concat([features, target], axis=1)

# Calculate the correlation matrix
correlation_matrix = all_data.corr()
# Save the entire correlation matrix to a file (e.g., CSV)
correlation_matrix.to_csv('correlation_matrix.csv', header=True)



from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import LabelEncoder
 # Encode the categorical target variable
label_encoder = LabelEncoder()
target_encoded = label_encoder.fit_transform(target)
mutual_info_scores = mutual_info_classif(features, target_encoded)
# Save mutual information scores to a CSV file
mutual_info_df = pd.DataFrame({'Feature': features.columns, 'Mutual_Info_Score': mutual_info_scores})
mutual_info_df.to_excel('mutual_info_scores.xlsx', index=False)

# Hyperparameter tuning using Grid Search
# from sklearn.model_selection import GridSearchCV
#
# param_grid = {
#     'n_estimators': [50, 100, 150],
#     'max_depth': [None, 10, 20],
#     'min_samples_split': [2, 5, 10],
#     'min_samples_leaf': [1, 2, 4]
# }
#
# grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5, scoring='accuracy')
# grid_search.fit(X_train, y_train)
#
# best_model = grid_search.best_estimator_
# print("Best Hyperparameters:", grid_search.best_params_)
#
# # Save the best trained model to a joblib file
# joblib.dump(best_model, 'best_football_prediction_model.joblib')


# Save the training data to an Excel file
# pd.DataFrame(X_train).to_excel('X_train.xlsx', index=False)
#
# # Make predictions on the test data
# predictions = best_model.predict(X_test)
#
# # Calculate accuracy on the test set
# accuracy = accuracy_score(y_test, predictions)
# print(f"Accuracy on the test set: {accuracy * 100:.2f}%")
#
# # Generate a classification report with zero_division parameter set to 1
# report = classification_report(y_test, predictions, zero_division=1)
# print("Classification Report:\n", report)
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
# #
# # # Save the model details to an Excel file
# # model_details = pd.DataFrame({'Feature': X_train.columns, 'Importance': best_model.feature_importances_})
# # model_details.to_excel('model_details.xlsx', index=False)