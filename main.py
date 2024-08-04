import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import missingno as msn
import warnings
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# Load datasets with error handling
try:
    df = pd.read_csv('CO2 Emissions_Canada.csv')
except FileNotFoundError:
    raise FileNotFoundError("The specified CSV file was not found.")
except pd.errors.EmptyDataError:
    raise ValueError("No data found in the CSV file.")
except pd.errors.ParserError:
    raise ValueError("Error parsing the CSV file.")

# Inspect the data
print(df.head())
print(df.info())
print(df.describe())

# Check column names
print("Column names before stripping:", df.columns.tolist())

# Clean column names (remove leading/trailing spaces)
df.columns = df.columns.str.strip()

# Check column names after cleaning
print("Column names after stripping:", df.columns.tolist())

# Ensure 'CO2 Emissions(g/km)' is a valid column name
if 'CO2 Emissions(g/km)' not in df.columns:
    raise KeyError("'CO2 Emissions(g/km)' not found in the DataFrame columns")

# Handle missing values
print("Missing values before imputation:\n", df.isnull().sum())
df.fillna(df.median(numeric_only=True), inplace=True)  # Impute missing values with median for numeric columns

# Check for any remaining missing values
print("Missing values after imputation:\n", df.isnull().sum())

# Drop irrelevant columns (if any)
# df.drop(columns=['Unnecessary_Column'], inplace=True)  # Adjust if you have unnecessary columns

# Convert categorical variables to numerical
df = pd.get_dummies(df, drop_first=True)

# Inspect the processed data
print(df.head())

# Define features and target
features = df.drop(columns=['CO2 Emissions(g/km)'])
target = df['CO2 Emissions(g/km)']

# Example feature engineering (modify as per your dataset)
# df['New_Feature'] = df['Feature1'] * df['Feature2']  # Example; adapt as needed

# Feature scaling
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features_scaled, target, test_size=0.2, random_state=42)

print(f"Training set size: {X_train.shape[0]}")
print(f"Testing set size: {X_test.shape[0]}")

# Initialize and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R^2 Score: {r2}")

# Cross-validation
cv_scores = cross_val_score(model, features_scaled, target, cv=5, scoring='r2')
print(f"Cross-Validation R^2 Scores: {cv_scores}")
print(f"Average Cross-Validation R^2 Score: {np.mean(cv_scores)}")

# Plot predictions vs actual values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.7)
plt.xlabel('Actual Emissions')
plt.ylabel('Predicted Emissions')
plt.title('Actual vs Predicted CO2 Emissions')
# Select numerical columns for pair plots
numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns

# Generate pair plots for numerical columns
sns.pairplot(df[numerical_cols])
plt.show()
