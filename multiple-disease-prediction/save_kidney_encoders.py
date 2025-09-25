# save_kidney_encoders.py
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import joblib
import os

df = pd.read_csv("datasets/kidney_disease.csv")
columns = ['age','bp','sg','al','su','rbc','pc','pcc','ba','bgr','bu']
kidney_df = df[columns].dropna()

os.makedirs("models", exist_ok=True)

cols_to_encode = ['rbc', 'pc', 'pcc', 'ba']
for c in cols_to_encode:
    le = LabelEncoder()
    le.fit(kidney_df[c])
    joblib.dump(le, f"models/{c}_le.pkl")
    print(f"Saved encoder for {c} -> models/{c}_le.pkl (classes: {list(le.classes_)})")
