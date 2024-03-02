# Random Forest algorithm:

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, multilabel_confusion_matrix, f1_score
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

# Load the CSV file into a DataFrame
df = pd.read_csv('output_selected_with_categorical_withoutMI.csv')

# Select the relevant columns for training and testing
features = df.drop('FullTimeResult', axis=1)
target = df['FullTimeResult']

# Convert categorical variables to numerical using one-hot encoding
features = pd.get_dummies(features)

# Split the data into training and testing sets
X_train = features.iloc[:3800, :]
y_train = target.iloc[:3800]

X_test = features.iloc[-380:, :]
y_test = target.iloc[-380:]


class_weights = compute_class_weight(
                                        class_weight = "balanced",
                                        classes = np.unique(y_train),
                                        y = y_train
                                    )
class_weights = dict(zip(np.unique(y_train), class_weights))
print(class_weights)

rfmodel = RandomForestClassifier(random_state=42, class_weight='balanced')
n_estimators = [50, 100, 150, 200, 500]
max_features = ['sqrt']
max_depth = [5, 10, 20]
max_depth.append(None)
min_samples_split = [2, 5, 10]
min_samples_leaf = [1, 2, 4]


# for n in n_estimators:
#     for depth in max_depth:
#         for split in min_samples_split:
#             for leaf in min_samples_leaf:
#                 rfmodel.min_samples_leaf = leaf
#                 rfmodel.min_samples_split = split
#                 rfmodel.max_depth = depth
#                 rfmodel.n_estimators=n
#                 rfmodel.fit(X_train, y_train)
#                 predictions = rfmodel.predict(X_test)
#                 accuracy = accuracy_score(y_test, predictions)
#                 precision = precision_score(y_test, predictions, average='weighted')
#                 f1score = f1_score(y_test, predictions, average='weighted')
#                 recall = recall_score(y_test, predictions, average='weighted')
#                 print(f"Accuracy on the test set: {accuracy * 100:.2f}%")
#                 print(f"precision on the test set: {precision * 100:.2f}%")
#                 print(f"f1score on the test set: {f1score * 100:.2f}%")
#                 print(f"recall on the test set: {recall * 100:.2f}%")
#                 print(f"n: {rfmodel.n_estimators}, max_depth: {rfmodel.max_depth}, min_samples_split: {rfmodel.min_samples_split}, min_samples_leaf {rfmodel.min_samples_leaf}")
#                 matrix = confusion_matrix(y_test, predictions)
#                 print("Confusion matrix:\n", matrix)


best_model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10, min_samples_leaf=2, min_samples_split=2, class_weight=class_weights)  # You can adjust n_estimators as needed
best_model.fit(X_train, y_train)
predictions = best_model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy on the test set: {accuracy * 100:.2f}%")
report = classification_report(y_test, predictions, zero_division=1)
print("Classification Report:\n", report)
matrix = confusion_matrix(y_test, predictions)
print("Confusion matrix:\n", matrix)
matrix = multilabel_confusion_matrix(y_test, predictions)
print("multilabel_confusion_matrix matrix:\n", matrix)
f1score = f1_score(y_test, predictions, average='weighted')
print(f"f1score on the test set: {f1score * 100:.2f}%")

# Create a DataFrame with details of predicted rows
predicted_df = X_test.copy()

predicted_df['True_Label'] = y_test
predicted_df['Predicted_Label'] = predictions

# Save the DataFrame to an XLS file
predicted_df.to_excel('predictions.xlsx', index=False)
print("Predictions saved to 'predictions.xlsx'")

importances = best_model.feature_importances_
print(importances)
