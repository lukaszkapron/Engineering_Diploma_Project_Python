import pandas as pd
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import LabelEncoder

# Load data from 'output.csv'
data = pd.read_csv('output_for_correlation_matrix.csv')

# # Label encode the categorical variable
# label_encoder = LabelEncoder()
# data['FullTimeResult'] = label_encoder.fit_transform(data['FullTimeResult'])
#
# # Separate continuous and categorical variables
# continuous_vars = data.drop('FullTimeResult', axis=1)
# categorical_var = data['FullTimeResult']
# continuous_vars = pd.get_dummies(continuous_vars)
# continuous_vars.to_excel('continuous_vars.xlsx', index=False)
# categorical_var.to_excel('categorical_var.xlsx', index=False)
#
# # Calculate mutual information
# mi_scores = mutual_info_regression(continuous_vars, categorical_var)
#
# # Create a DataFrame to display the results
# mi_results = pd.DataFrame({'Feature': continuous_vars.columns, 'Mutual_Information': mi_scores})
# mi_results = mi_results.sort_values(by='Mutual_Information', ascending=False)
#
# # Display and save the results
# print(mi_results)
#
# # Save the results to a new CSV file
# mi_results.to_csv('mutual_information_results.csv', index=False)







# # Concatenate the features and target into a single DataFrame
# all_data = pd.concat([features, target], axis=1)
#
# # # Calculate the correlation matrix
# # correlation_matrix = features.corr()
#
# corr_matrix = features.corr().abs()
#
# # Select upper triangle of correlation matrix
# upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
#
# # Find features with correlation greater than 0.95
# to_drop = [column for column in upper.columns if any(upper[column] > 0.8)]
#
# # Drop features
# features.drop(to_drop, axis=1, inplace=True)
#
# # Save the entire correlation matrix to a file (e.g., CSV)
# #corr_matrix.to_csv('correlation_matrix.csv', header=True)
# print(features)
# features.to_excel('features_after_remove.xlsx', header=True)








import pandas as pd

# Assuming your dataset is stored in a DataFrame called df
# Replace 'your_dataset.csv' with the actual file path or use your data directly
# df = pd.read_csv('your_dataset.csv')

# If your dataset is already loaded, you can proceed with the following code

# Assuming your categorical columns are stored in a list called categorical_columns
# and continuous columns are stored in a list called continuous_columns
categorical_columns = ['Id', 'HomeTeamIsFromBigSix', 'HomeTeamIsNewInPL', 'HomeTeamLastH2HMatchResultHome','AwayTeamIsFromBigSix', 'AwayTeamIsNewInPL', 'AwayTeamLastH2HMatchResultAway', 'FullTimeResult']
continuous_columns = [col for col in data.columns if col not in categorical_columns]

# Calculate correlation matrix
correlation_matrix = data[continuous_columns].corr().abs()
# Function to remove '.1' from column labels
# Function to clean column labels
def clean_labels(column):
    parts = column.split('.')
    # Check if the last part is a digit and if it is, only take the first part
    if parts[-1].isdigit():
        return '.'.join(parts[:-1])
    else:
        return column
correlation_matrix.columns = correlation_matrix.columns.map(clean_labels)

correlation_matrix.to_excel('correlation_matrix.xlsx', header=True)

import matplotlib.pyplot as plt
import seaborn as sns


def custom_labels(column, substring_to_remove='.1'):
    # Replace the specific substring with an empty string
    modified_label = column.replace(substring_to_remove, '')

    return modified_label
# Define the number of rows and columns for the subplots
rows, cols = 2, 2

# Calculate the size of each quadrant
split_rows = len(correlation_matrix) // rows
split_cols = len(correlation_matrix) // cols

import numpy as np
#Create individual plots for each sub-quadrant
for i in range(rows):
    for j in range(cols):
        start_row, end_row = i * split_rows, (i + 1) * split_rows
        start_col, end_col = j * split_cols, (j + 1) * split_cols

        # Extract the sub-quadrant
        sub_matrix = correlation_matrix.iloc[start_row:end_row, start_col:end_col]

        # Create a figure for each sub-quadrant
        fig, ax = plt.subplots(figsize=(64, 64))  # Increase figure size
        sns.heatmap(sub_matrix, annot=False, cmap='coolwarm', fmt=".2f", linewidths=.5, square=True, cbar=True)

        # ax.set_xticks(np.arange(len(sub_matrix.columns)))
        # ax.set_xticklabels(sub_matrix.columns)
        # Rotate labels
        # plt.xticks(rotation=45, ha='right')  # Adjust rotation and horizontal alignment as needed
        # plt.yticks(rotation=0)


        ax.set_xticklabels(sub_matrix.columns.map(custom_labels), ha='right')
        ax.set_yticklabels(sub_matrix.index.map(custom_labels))



        plt.tick_params(axis='both', which='both', labelsize=6)

        # Save the plot to a file (adjust the filename as needed)
        plt.savefig(f'sub_quadrant_{i + 1}_{j + 1}.png')
        # Show the plot
        plt.show()




# import matplotlib.pyplot as plt
# import seaborn as sns
#
# fig, ax = plt.subplots(figsize=(64, 64))  # Increase figure size
# sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', fmt=".2f", linewidths=.5, square=True, cbar=True)
#
#
# ax.set_xticklabels(correlation_matrix.columns, ha='right')
# ax.set_yticklabels(correlation_matrix.index)
#
#
# plt.tick_params(axis='both', which='both', labelsize=6)
#
# plt.show()
