import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load the CSV file into a DataFrame
df = pd.read_csv('output_selected_with_categorical_withoutMI.csv')

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

# # Save training data to Excel
# X_train.to_excel('X_train.xlsx', index=False)
# y_train.to_excel('y_train.xlsx', index=False)
#
# # Save test data to Excel
# X_test.to_excel('X_test.xlsx', index=False)
# y_test.to_excel('y_test.xlsx', index=False)



rfmodel = RandomForestClassifier(random_state=42)
n_estimators = [50, 100, 150, 200, 500]
max_features = ['sqrt']
max_depth = [5, 10, 20]
max_depth.append(None)
min_samples_split = [2, 5, 10]
min_samples_leaf = [1, 2, 4]


for n in n_estimators:
    for depth in max_depth:
        for split in min_samples_split:
            for leaf in min_samples_leaf:
                rfmodel.min_samples_leaf = leaf
                rfmodel.min_samples_split = split
                rfmodel.max_depth = depth
                rfmodel.n_estimators=n
                rfmodel.fit(X_train, y_train)
                predictions = rfmodel.predict(X_test)
                accuracy = accuracy_score(y_test, predictions)
                print(f"Accuracy on the test set: {accuracy * 100:.2f}%")
                print(f"n: {rfmodel.n_estimators}, max_depth: {rfmodel.max_depth}, min_samples_split: {rfmodel.min_samples_split}, min_samples_leaf {rfmodel.min_samples_leaf}")

# best_model = RandomForestClassifier(n_estimators=50, random_state=42, max_depth=None, min_samples_leaf=4, min_samples_split=10)  # You can adjust n_estimators as needed
# best_model.fit(X_train, y_train)
# predictions = best_model.predict(X_test)
# accuracy = accuracy_score(y_test, predictions)
# print(f"Accuracy on the test set: {accuracy * 100:.2f}%")
# report = classification_report(y_test, predictions, zero_division=1)
# print("Classification Report:\n", report)






#Hyperparameter tuning using Grid Search
# from sklearn.model_selection import GridSearchCV
#
# import numpy as np
# # Number of trees in random forest
# n_estimators = [10,100,1000]
# # Number of features to consider at every split
# max_features = ['sqrt']
# # Maximum number of levels in tree
# max_depth = [5,10,20]
# max_depth.append(None)
# # Minimum number of samples required to split a node
# min_samples_split = [2, 5, 10]
# # Minimum number of samples required at each leaf node
# min_samples_leaf = [1, 2, 4]
# # Method of selecting samples for training each tree
# bootstrap = [True, False]
# # Create the random grid
# param_grid = {'n_estimators': n_estimators,
#                'max_features': max_features,
#                'max_depth': max_depth,
#                'min_samples_split': min_samples_split,
#                'min_samples_leaf': min_samples_leaf,
#                'bootstrap': bootstrap}
# print(param_grid)

# param_grid = {
#     'n_estimators': [25, 50, 100, 150],
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





# import itertools
# from sklearn.model_selection import cross_val_score
# import numpy as np
# param_grid = {
#     'n_estimators': [50, 100, 150, 200, 500],
#     'max_depth': [None, 10, 20],
#     'min_samples_split': [2, 5, 10],
#     'min_samples_leaf': [1, 2, 4]
# }
#
#
# # Wszystkie kombinacje parametrów
# param_combinations = list(itertools.product(*param_grid.values()))
#
# # Definicja modelu Random Forest
# random_forest = RandomForestClassifier()
#
# # Przeszukiwanie wszystkich kombinacji parametrów
# for params in param_combinations:
#     param_dict = dict(zip(param_grid.keys(), params))
#     random_forest.set_params(**param_dict)
#
#     # Użycie cross_val_score do oceny modelu
#     scores = cross_val_score(random_forest, X_train, y_train, cv=5)
#     mean_score = np.mean(scores)
#
#     # Wyświetlanie kombinacji parametrów i skuteczności
#     print(f"Parametry: {param_dict}, Skuteczność: {mean_score}")