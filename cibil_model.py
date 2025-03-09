# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.preprocessing import LabelEncoder
# from sklearn.metrics import mean_absolute_error, r2_score

# # Load dataset
# df = pd.read_csv("./synthetic_credit_dataset.csv")  # Ensure correct path

# # Drop non-numeric columns if they exist
# df = df.drop(columns=['BusinessID'], errors='ignore')

# # Identify and encode categorical columns
# categorical_cols = df.select_dtypes(include=['object']).columns
# label_encoders = {}
# for col in categorical_cols:
#     le = LabelEncoder()
#     df[col] = le.fit_transform(df[col])
#     label_encoders[col] = le  # Store encoders for later use

# # Handle missing values
# df.fillna(df.mean(), inplace=True)

# # Define target and features
# target_column = 'Credit_Score'
# X = df.drop(columns=[target_column])
# y = df[target_column]

# # Train model
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# rf_model = RandomForestRegressor(n_estimators=50, random_state=42)
# rf_model.fit(X_train, y_train)

# # Model evaluation
# y_pred = rf_model.predict(X_test)
# mae = mean_absolute_error(y_test, y_pred)
# r2 = r2_score(y_test, y_pred)
# print(f"🔹 Mean Absolute Error (MAE): {mae:.2f}")
# print(f"🔹 R² Score: {r2:.4f}")

# # Prediction function
# def predict_cibil(features):
#     return np.clip(rf_model.predict([features])[0], 300, 900)  # Ensure within valid range


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score

# Load dataset
df = pd.read_csv("./synthetic_credit_dataset.csv")  # Ensure correct path

# Drop non-numeric columns if they exist
df = df.drop(columns=['BusinessID'], errors='ignore')

# Identify and encode categorical columns
categorical_cols = df.select_dtypes(include=['object']).columns
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le  # Store encoders for later use

# Handle missing values
df.fillna(df.mean(), inplace=True)

# Define target and features
target_column = 'Credit_Score'
X = df.drop(columns=[target_column])
y = df[target_column]

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
rf_model = RandomForestRegressor(n_estimators=50, random_state=42)
rf_model.fit(X_train, y_train)

# Model evaluation
y_pred = rf_model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"🔹 Mean Absolute Error (MAE): {mae:.2f}")
print(f"🔹 R² Score: {r2:.4f}")

# Prediction function
def predict_cibil(features):
    return np.clip(rf_model.predict([features])[0], 300, 900)  # Ensure within valid range