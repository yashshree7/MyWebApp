import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib

# Define the data
data = pd.read_csv("diabetes_original.csv")

# Load data into a DataFrame
df = pd.DataFrame(data)

# Replace zeros in specific columns with the median of the column
columns_to_replace = ['BloodPressure', 'SkinThickness', 'Insulin']
for column in columns_to_replace:
    df[column] = df[column].replace(0, df[column].median())

# Split data into features and target variable
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train a Random Forest Classifier
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Save the model and scaler
joblib.dump(model, 'model.pkl')
joblib.dump(scaler, 'scaler.pkl')
