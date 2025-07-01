import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

from sklearn.preprocessing import LabelEncoder
import joblib
import os

# Load dataset
df = pd.read_csv("backend/data/calories.csv")

# Encode 'Gender' column (male -> 1, female -> 0)
gender_encoder = LabelEncoder()
df["Gender"] = gender_encoder.fit_transform(df["Gender"])

# Separate features (X) and target (y)
X = df.drop(columns=["User_ID", "Calories"])  # Keep Gender
y = df["Calories"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)

model.fit(X_train, y_train)

# Save the trained model
os.makedirs("backend/model", exist_ok=True)
joblib.dump(model, "backend/model/calories_model.pkl")

# Save the encoder (optional but useful for prediction)
joblib.dump(gender_encoder, "backend/model/gender_encoder.pkl")

print("✅ Model trained and saved to backend/model/calories_model.pkl")
print("✅ Gender encoder saved to backend/model/gender_encoder.pkl")
