from ucimlrepo import fetch_ucirepo 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt
import xgboost as xgb
print("Starting the task...")
# Fetch dataset
covertype = fetch_ucirepo(id=31)
X = covertype.data.features
y = covertype.data.targets
print("Dataset fetched successfully.")
# Data cleaning & preprocessing
# If y is a dataframe, convert to Series
if isinstance(y, pd.DataFrame):
    y = y.iloc[:, 0]
print("Data shape:", X.shape, y.shape)
# Encode categorical variables if any
for col in X.select_dtypes(include=['object']).columns:
    X[col] = LabelEncoder().fit_transform(X[col])
print("Categorical variables encoded.")
# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train/test split
if y.min() != 0:
    y = y - y.min()
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# Random Forest
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

print("Random Forest Classification Report:")
print(classification_report(y_test, y_pred_rf))

# XGBoost
xgb_model = xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='mlogloss')
xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)

print("XGBoost Classification Report:")
print(classification_report(y_test, y_pred_xgb))

# Confusion Matrix for Random Forest
plt.figure(figsize=(8,6))
ConfusionMatrixDisplay.from_estimator(rf, X_test, y_test, cmap='Blues')
plt.title("Random Forest Confusion Matrix")
plt.show()

# Confusion Matrix for XGBoost
plt.figure(figsize=(8,6))
ConfusionMatrixDisplay.from_estimator(xgb_model, X_test, y_test, cmap='Greens')
plt.title("XGBoost Confusion Matrix")
plt.show()

# Feature Importance (Random Forest)
importances_rf = rf.feature_importances_
indices_rf = importances_rf.argsort()[::-1]
plt.figure(figsize=(10,5))
plt.title("Random Forest Feature Importances")
plt.bar(range(X.shape[1]), importances_rf[indices_rf])
plt.xticks(range(X.shape[1]), X.columns[indices_rf], rotation=90)
plt.tight_layout()
plt.show()

# Feature Importance (XGBoost)
importances_xgb = xgb_model.feature_importances_
indices_xgb = importances_xgb.argsort()[::-1]
plt.figure(figsize=(10,5))
plt.title("XGBoost Feature Importances")
plt.bar(range(X.shape[1]), importances_xgb[indices_xgb])
plt.xticks(range(X.shape[1]), X.columns[indices_xgb], rotation=90)
plt.tight_layout()
plt.show()

