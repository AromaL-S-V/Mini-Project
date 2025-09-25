# train_model_regularized.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import pickle
import os

# Create models directory if not exists
os.makedirs("models", exist_ok=True)

# ------------------ Utility function ------------------
def build_model(input_dim):
    model = Sequential([
        Dense(16, activation='relu', input_shape=(input_dim,)),
        Dropout(0.3),
        Dense(8, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def evaluate_model(model, X_train, y_train, X_test, y_test, name):
    # Evaluate
    train_loss, train_acc = model.evaluate(X_train, y_train, verbose=0)
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\n✅ {name} - Train Accuracy: {train_acc:.4f}, Test Accuracy: {test_acc:.4f}")

    # Predictions
    y_pred = (model.predict(X_test) > 0.5).astype(int)
    print(f"\n📊 {name} - Classification Report")
    print(classification_report(y_test, y_pred, digits=4))

    print(f"🔎 {name} - Confusion Matrix")
    print(confusion_matrix(y_test, y_pred))


# ------------------ Diabetes ------------------
print("\n🔹 Training Diabetes Model...")
diabetes_df = pd.read_csv("datasets/diabetes.csv")
X_diabetes = diabetes_df.drop("Outcome", axis=1)
y_diabetes = diabetes_df["Outcome"]

scaler_diabetes = StandardScaler()
X_diabetes_scaled = scaler_diabetes.fit_transform(X_diabetes)

X_train_d, X_test_d, y_train_d, y_test_d = train_test_split(
    X_diabetes_scaled, y_diabetes, test_size=0.2, random_state=42
)

model_diabetes = build_model(X_train_d.shape[1])
es = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

model_diabetes.fit(X_train_d, y_train_d, epochs=100, validation_data=(X_test_d, y_test_d),
                   callbacks=[es], verbose=0)

evaluate_model(model_diabetes, X_train_d, y_train_d, X_test_d, y_test_d, "Diabetes")

# Save
model_diabetes.save("models/diabetes_model.h5")
with open("models/diabetes_scaler.pkl", "wb") as f:
    pickle.dump(scaler_diabetes, f)


# ------------------ Heart ------------------
print("\n🔹 Training Heart Model...")
heart_df = pd.read_csv("datasets/heart.csv")
X_heart = heart_df.drop("target", axis=1)
y_heart = heart_df["target"]

scaler_heart = StandardScaler()
X_heart_scaled = scaler_heart.fit_transform(X_heart)

X_train_h, X_test_h, y_train_h, y_test_h = train_test_split(
    X_heart_scaled, y_heart, test_size=0.2, random_state=42
)

model_heart = build_model(X_train_h.shape[1])
es = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

model_heart.fit(X_train_h, y_train_h, epochs=100, validation_data=(X_test_h, y_test_h),
                callbacks=[es], verbose=0)

evaluate_model(model_heart, X_train_h, y_train_h, X_test_h, y_test_h, "Heart")

# Save
model_heart.save("models/heart_model.h5")
with open("models/heart_scaler.pkl", "wb") as f:
    pickle.dump(scaler_heart, f)


# ------------------ Kidney ------------------
print("\n🔹 Training Kidney Model...")
kidney_df = pd.read_csv("datasets/kidney_disease.csv")

columns = ['age', 'bp', 'sg', 'al', 'su', 'rbc', 'pc', 'pcc', 'ba', 'bgr', 'bu']
kidney_df = kidney_df[columns].dropna()

# Encode categorical columns
for col in ['rbc', 'pc', 'pcc', 'ba']:
    le = LabelEncoder()
    kidney_df[col] = le.fit_transform(kidney_df[col])

# Create target
kidney_df['target'] = kidney_df.apply(lambda row: 1 if row['al'] > 0 or row['su'] > 0 else 0, axis=1)

X_kidney = kidney_df[columns]
y_kidney = kidney_df['target']

scaler_kidney = StandardScaler()
X_kidney_scaled = scaler_kidney.fit_transform(X_kidney)

X_train_k, X_test_k, y_train_k, y_test_k = train_test_split(
    X_kidney_scaled, y_kidney, test_size=0.2, random_state=42
)

model_kidney = build_model(X_train_k.shape[1])
es = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Handle imbalance
class_weights = compute_class_weight('balanced', classes=np.unique(y_train_k), y=y_train_k)
class_weights_dict = dict(enumerate(class_weights))

model_kidney.fit(X_train_k, y_train_k, epochs=100, validation_data=(X_test_k, y_test_k),
                 callbacks=[es], class_weight=class_weights_dict, verbose=0)

evaluate_model(model_kidney, X_train_k, y_train_k, X_test_k, y_test_k, "Kidney")

# Save
model_kidney.save("models/kidney_model.h5")
with open("models/kidney_scaler.pkl", "wb") as f:
    pickle.dump(scaler_kidney, f)

print("\n🎉 All models trained, evaluated, and saved successfully with regularization.")
