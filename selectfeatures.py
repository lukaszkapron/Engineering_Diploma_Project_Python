import pandas as pd

data = pd.read_csv('output_for_selection.csv')

correlation_matrix = data.corr().abs()

correlation_matrix.to_excel('correlation_matrix_for_selection.xlsx', header=True)

from sklearn.feature_selection import mutual_info_classif
import pandas as pd
import numpy as np

df = pd.read_csv('output.csv')
np.random.seed(42)

# Assuming df is your DataFrame with continuous and categorical variables
X = df.drop("FullTimeResult", axis=1)  # Features
X = X.drop("Id", axis=1)  # Features
X = X.drop("HomeTeamIsFromBigSix", axis=1)  # Features
X = X.drop("AwayTeamIsFromBigSix", axis=1)  # Features
X = X.drop("HomeTeamIsNewInPL", axis=1)  # Features
X = X.drop("AwayTeamIsNewInPL", axis=1)  # Featurese
X = X.drop("HomeTeamLastH2HMatchResultHome", axis=1)  # Features
X = X.drop("AwayTeamLastH2HMatchResultAway", axis=1)  # Features

y = df["FullTimeResult"]  # Target variable

categorical_cols = X.select_dtypes(include=['object']).columns
X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

# Calculate mutual information for each feature
mutual_info_scores = mutual_info_classif(X_encoded, y)


# Print or visualize the scores
for column, score in zip(X_encoded.columns, mutual_info_scores):
    print(f"Mutual Information between 'target' and '{column}': {score}")

# Create a DataFrame with feature names and their mutual information scores
mi_df = pd.DataFrame({'Feature': X_encoded.columns, 'Mutual_Information': mutual_info_scores})

# Save the DataFrame to a CSV file
mi_df.to_excel('mutual_information_scores.xlsx', index=False)

to_drop = set()  # Use a set to store columns to drop to avoid duplicates

for i, col1 in enumerate(X.columns):
    for j, col2 in enumerate(X.columns):
        if i < j and correlation_matrix.loc[col1, col2] > 0.8:
            if mutual_info_scores[i] < mutual_info_scores[j]:
                to_drop.add(col1)
            else:
                to_drop.add(col2)

# Step 5: Drop selected features
data_filtered = data.drop(to_drop, axis=1)

# Step 6: Save the new DataFrame to an Excel file
data_filtered.to_excel('filtered_data.xlsx', header=True, index=False)


corr = data_filtered.corr().abs()
import matplotlib.pyplot as plt
import seaborn as sns

fig, ax = plt.subplots(figsize=(64, 64))  # Increase figure size
sns.heatmap(corr, annot=False, cmap='coolwarm', fmt=".2f", linewidths=.5, square=True, cbar=True)


ax.set_xticklabels(corr.columns, ha='right')
ax.set_yticklabels(corr.index)



plt.tick_params(axis='both', which='both', labelsize=6)

plt.show()
