# Visualize correlation matrix:

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data from 'output.csv'
data = pd.read_csv('output_for_correlation_matrix.csv')


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

def custom_labels(column, substring_to_remove='.1'):
    # Replace the specific substring with an empty string
    modified_label = column.replace(substring_to_remove, '')

    return modified_label
# Define the number of rows and columns for the subplots
rows, cols = 2, 2

# Calculate the size of each quadrant
split_rows = len(correlation_matrix) // rows
split_cols = len(correlation_matrix) // cols

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

        ax.set_xticklabels(sub_matrix.columns.map(custom_labels), ha='right')
        ax.set_yticklabels(sub_matrix.index.map(custom_labels))

        plt.tick_params(axis='both', which='both', labelsize=6)

        # Save the plot to a file (adjust the filename as needed)
        plt.savefig(f'sub_quadrant_{i + 1}_{j + 1}.png')
