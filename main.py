# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler, LabelEncoder
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import classification_report
# from sklearn.metrics import accuracy_score
#
# matches = pd.read_csv("output.csv", index_col=0)
#
# label_encoder = LabelEncoder()
#
# for column in matches.columns:
#     if matches[column].dtype == 'object':  # Check if the column contains strings
#         matches[column] = label_encoder.fit_transform(matches[column])
#
#
# print(matches.head)
#
# train = matches.take([0, 3419])
# test = matches.take([3420, 3799])
# #train, test = train_test_split(matches, test_size=0.2, random_state=1)
#
#
# predictors = ["HomeTeam", "HomeIsFromBigSix","HomeIsNewInPL","HomeRedCardsInLastMatch","AwayTeam","AwayIsFromBigSix", "AwayIsNewInPL","AwayRedCardsInLastMatch"]
# rf = RandomForestClassifier(n_estimators=200, random_state=42)
#
# rf.fit(train[predictors], train["FullTimeResult"])
# print(train[predictors])
# print(test[predictors])
#
# preds = rf.predict(test[predictors])
# error = classification_report(test["FullTimeResult"], preds)
# print(error)
#
# accuracy = accuracy_score(test["FullTimeResult"], preds)
# print("Accuracy Score:", accuracy)
#
# print("Unique labels in training data:", train["FullTimeResult"].unique())
# print("Unique labels in test data:", test["FullTimeResult"].unique())
#
#
#
#
# #
# #
# #
# #
# #
# # # # X to dane liczbowe - wszystkie wiersze, wszystkie kolumny poza ostatnią
# # # X = matches.values
# # # # Y to etykiety, czyli przynależność do danego gatunku irysu
# # # y = matches["FTR"]
# # #
# # # # konwersja etykiet w formie napisów na liczby
# # # le = LabelEncoder()
# # # y = le.fit_transform(y)
# # #
# # # # Wyświetlanie X i y
# # # print(X)
# # # print(y)
# # #
# # # # Normalizacja danych liczbowych
# # # scaler = StandardScaler()
# # # scaler.fit(X)
# # # X = scaler.transform(X)
# # #
# # # # Dzielimy zbiór na 20% testowych i 80% trenujących
# # # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
# # #
# # # # Wyświetlenie jak wyglądają teraz dane testowe po normalizacja
# # # print(X_test)
# # #
# # # # Klasyfikacja używając metody RF dla 200 drzew
# # # rf = RandomForestClassifier(n_estimators=200, random_state=42)
# # #
# # # # Dostarczenie klasyfikatorowi danych trenujących
# # # rf.fit(X_train, y_train)
# # #
# # # # Predykcja wyników na podstawie danych testowych
# # # y_pred = rf.predict(X_test)
# # #
# # # print(classification_report(y_test, y_pred))
#
#
#
#
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load the CSV file into a DataFrame
df = pd.read_csv('output.csv')

# Select the relevant columns for training and testing
features = ['HomeTeam', 'AwayTeam', 'HomeIsFromBigSix', 'HomeIsNewInPL', 'HomeRedCardsInLastMatch',
            'AwayIsFromBigSix', 'AwayIsNewInPL', 'AwayRedCardsInLastMatch']
target = 'FullTimeResult'

# Split the data into training (first 3420 rows) and testing (last 380 rows)
train_data = df[:3420]
test_data = df[-380:]

# Check unique values in 'HomeTeam' and 'AwayTeam' columns for both training and test sets
train_home_teams = set(train_data['HomeTeam'].unique())
train_away_teams = set(train_data['AwayTeam'].unique())
test_home_teams = set(test_data['HomeTeam'].unique())
test_away_teams = set(test_data['AwayTeam'].unique())

# Find missing categories in the test set and create a list of all unique categories
unique_categories = list(train_home_teams.union(train_away_teams, test_home_teams, test_away_teams))

# Combine 'HomeTeam' and 'AwayTeam' columns into one column for one-hot encoding
combined_data = pd.concat([train_data[['HomeTeam', 'AwayTeam']], test_data[['HomeTeam', 'AwayTeam']]])

# Perform one-hot encoding on the combined data
combined_encoded = pd.get_dummies(combined_data, columns=['HomeTeam', 'AwayTeam'], drop_first=True)

# Split the encoded data back into training and test sets
X_train_encoded = combined_encoded[:len(train_data)]
X_test_encoded = combined_encoded[len(train_data):]

# Split the data into features and target
y_train = train_data[target]
y_test = test_data[target]

# Initialize and train a Random Forest classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)  # You can adjust n_estimators as needed
model.fit(X_train_encoded, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test_encoded)

# Calculate accuracy on the test set
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy on the test set: {accuracy * 100:.2f}%")

# Generate a classification report with zero_division parameter set to 1
report = classification_report(y_test, y_pred, zero_division=1)
print("Classification Report:\n", report)

# Create a DataFrame with details of predicted rows
predicted_df = test_data.copy()
predicted_df['True_Label'] = y_test
predicted_df['Predicted_Label'] = y_pred

# Save the DataFrame to an XLS file
predicted_df.to_excel('predictions.xlsx', index=False)
print("Predictions saved to 'predictions.xlsx'")
