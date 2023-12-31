from sklearn.feature_selection import mutual_info_classif
import pandas as pd
import numpy as np

df = pd.read_csv('output.csv')
np.random.seed(42)

# Assuming df is your DataFrame with continuous and categorical variables
X = df.drop("FullTimeResult", axis=1)  # Features
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