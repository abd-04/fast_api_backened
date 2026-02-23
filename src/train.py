import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

# 1️⃣ Load data
df = pd.read_csv("../data/diabetes.csv")

# 2️⃣ Handle zero values (example)
#cols_with_zero = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
#df[cols_with_zero] = df[cols_with_zero].replace(0, np.nan)
#df[cols_with_zero] = df[cols_with_zero].fillna(df[cols_with_zero].median())

# 3️⃣ Split
#X = df.drop("Outcome", axis=1)
X=df.drop("Outcome", axis=1)
#X=df[['Age','Glucose','BMI','Insulin','Pregnancies']]
y = df["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4️⃣ Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5️⃣ Train
model = LogisticRegression(class_weight="balanced", max_iter=1000)
model.fit(X_train_scaled, y_train)

# 6️⃣ Evaluate
probs = model.predict_proba(X_test_scaled)[:, 1]

threshold = 0.48  # try 0.45, 0.4, 0.35
predictions = (probs >= threshold).astype(int)

print("Accuracy:", accuracy_score(y_test, predictions))
print("Recall:", recall_score(y_test, predictions))
print("Precision:", precision_score(y_test, predictions))
print("F1:", f1_score(y_test, predictions))

# 7️⃣ Save artifacts
joblib.dump(model, "../models/diabetes_model.pkl")
joblib.dump(scaler, "../models/scaler.pkl")