import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import pickle
import os

# Create models directory if not exists
os.makedirs("models", exist_ok=True)

# ------------------ Diabetes Dataset ------------------
diabetes_df = pd.read_csv("datasets/diabetes.csv")
X_diabetes = diabetes_df.drop("Outcome", axis=1)
y_diabetes = diabetes_df["Outcome"]

scaler_diabetes = StandardScaler()
X_diabetes_scaled = scaler_diabetes.fit_transform(X_diabetes)

X_train_d, X_test_d, y_train_d, y_test_d = train_test_split(
    X_diabetes_scaled, y_diabetes, test_size=0.2, random_state=42
)

model_diabetes = Sequential([
    Dense(16, activation='relu', input_shape=(X_train_d.shape[1],)),
    Dense(8, activation='relu'),
    Dense(1, activation='sigmoid')
])

model_diabetes.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model_diabetes.fit(X_train_d, y_train_d, epochs=100, verbose=0)
model_diabetes.save("models/diabetes_model.h5")
with open("models/diabetes_scaler.pkl", "wb") as f:
    pickle.dump(scaler_diabetes, f)

# ------------------ Heart Dataset ------------------
heart_df = pd.read_csv("datasets/heart.csv")
X_heart = heart_df.drop("target", axis=1)
y_heart = heart_df["target"]

scaler_heart = StandardScaler()
X_heart_scaled = scaler_heart.fit_transform(X_heart)

X_train_h, X_test_h, y_train_h, y_test_h = train_test_split(
    X_heart_scaled, y_heart, test_size=0.2, random_state=42
)

model_heart = Sequential([
    Dense(16, activation='relu', input_shape=(X_train_h.shape[1],)),
    Dense(8, activation='relu'),
    Dense(1, activation='sigmoid')
])

model_heart.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model_heart.fit(X_train_h, y_train_h, epochs=100, verbose=0)
model_heart.save("models/heart_model.h5")
with open("models/heart_scaler.pkl", "wb") as f:
    pickle.dump(scaler_heart, f)

# ------------------ Kidney Dataset (selected columns) ------------------
kidney_df = pd.read_csv("datasets/kidney_disease.csv")

# Columns to use
columns = ['age', 'bp', 'sg', 'al', 'su', 'rbc', 'pc', 'pcc', 'ba', 'bgr', 'bu']

# Drop rows with missing important values
kidney_df = kidney_df[columns].dropna()

# Encode categorical columns
for col in ['rbc', 'pc', 'pcc', 'ba']:
    le = LabelEncoder()
    kidney_df[col] = le.fit_transform(kidney_df[col])

# Create target: 1 if al > 0 or su > 0
kidney_df['target'] = kidney_df.apply(lambda row: 1 if row['al'] > 0 or row['su'] > 0 else 0, axis=1)

X_kidney = kidney_df[columns]
y_kidney = kidney_df['target']

scaler_kidney = StandardScaler()
X_kidney_scaled = scaler_kidney.fit_transform(X_kidney)

X_train_k, X_test_k, y_train_k, y_test_k = train_test_split(
    X_kidney_scaled, y_kidney, test_size=0.2, random_state=42
)

model_kidney = Sequential([
    Dense(16, activation='relu', input_shape=(X_train_k.shape[1],)),
    Dense(8, activation='relu'),
    Dense(1, activation='sigmoid')
])

model_kidney.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model_kidney.fit(X_train_k, y_train_k, epochs=100, verbose=0)
model_kidney.save("models/kidney_model.h5")
with open("models/kidney_scaler.pkl", "wb") as f:
    pickle.dump(scaler_kidney, f)

print("✅ All models and scalers saved successfully.")
