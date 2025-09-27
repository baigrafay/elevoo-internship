import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
data = pd.read_csv("Datasets/StudentPerformanceFactors.csv")

# Data cleaning
print(data.info())
print(data.describe())
print(data.isnull().sum())

# Basic visualization
sns.scatterplot(x='Hours_Studied', y='Exam_Score', data=data)
plt.title('Study Hours vs Exam Score')
plt.show()

# Feature selection
X = data[['Hours_Studied']]
y = data['Exam_Score']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Visualize predictions
plt.scatter(X_test, y_test, color='blue', label='Actual')
plt.scatter(X_test, y_pred, color='red', label='Predicted')
plt.xlabel('Study Hours')
plt.ylabel('Exam Score')
plt.legend()
plt.title('Actual vs Predicted Exam Scores')
plt.show()

# Evaluate model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse:.2f}")
print(f"R^2 Score: {r2:.2f}")

## Bonuses
from sklearn.preprocessing import PolynomialFeatures

# Polynomial Regression (degree 2)
poly = PolynomialFeatures(degree=2)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

poly_model = LinearRegression()
poly_model.fit(X_train_poly, y_train)
y_pred_poly = poly_model.predict(X_test_poly)

# Visualize polynomial predictions
plt.scatter(X_test, y_test, color='blue', label='Actual')
plt.scatter(X_test, y_pred_poly, color='green', label='Poly Predicted')
plt.xlabel('Study Hours')
plt.ylabel('Exam Score')
plt.legend()
plt.title('Actual vs Polynomial Predicted Exam Scores')
plt.show()

# Evaluate polynomial model
mse_poly = mean_squared_error(y_test, y_pred_poly)
r2_poly = r2_score(y_test, y_pred_poly)
print(f"Polynomial Mean Squared Error: {mse_poly:.2f}")
print(f"Polynomial R^2 Score: {r2_poly:.2f}")

# Encode categorical variables if needed
data_encoded = data.copy()
if data_encoded['Gender'].dtype == 'object':
    data_encoded['Gender'] = LabelEncoder().fit_transform(data_encoded['Gender'])

if data_encoded['Motivation_Level'].dtype == 'object':
    data_encoded['Motivation_Level'] = LabelEncoder().fit_transform(data_encoded['Motivation_Level'])    

X_multi = data_encoded[['Hours_Studied', 'Sleep_Hours', 'Gender' , 'Attendance' , 'Motivation_Level' , 'Tutoring_Sessions']]
y_multi = data_encoded['Exam_Score']

# Split dataset
X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(X_multi, y_multi, test_size=0.2, random_state=42)

# Linear Regression with multiple features
model_multi = LinearRegression()
model_multi.fit(X_train_m, y_train_m)
y_pred_m = model_multi.predict(X_test_m)

# Polynomial Regression with multiple features (degree 2)
poly_m = PolynomialFeatures(degree=2)
X_train_poly_m = poly_m.fit_transform(X_train_m)
X_test_poly_m = poly_m.transform(X_test_m)

poly_model_m = LinearRegression()
poly_model_m.fit(X_train_poly_m, y_train_m)
y_pred_poly_m = poly_model_m.predict(X_test_poly_m)

# Evaluate models
mse_m = mean_squared_error(y_test_m, y_pred_m)
r2_m = r2_score(y_test_m, y_pred_m)
mse_poly_m = mean_squared_error(y_test_m, y_pred_poly_m)
r2_poly_m = r2_score(y_test_m, y_pred_poly_m)

print("\n--- Multiple Feature Linear Regression ---")
print(f"Mean Squared Error: {mse_m:.2f}")
print(f"R^2 Score: {r2_m:.2f}")

print("\n--- Multiple Feature Polynomial Regression ---")
print(f"Polynomial Mean Squared Error: {mse_poly_m:.2f}")
print(f"Polynomial R^2 Score: {r2_poly_m:.2f}")

# Compare predictions
plt.figure(figsize=(10, 5))
plt.scatter(range(len(y_test)), y_test, color='blue', label='Actual')
plt.scatter(range(len(y_test)), y_pred, color='green', label='Linear Predicted')
plt.scatter(range(len(y_test)), y_pred_poly, color='red', label='Poly Predicted', alpha=0.6)
plt.legend()
plt.title("Prediction Comparison")
plt.xlabel("Sample")
plt.ylabel("Score")
plt.show()


# Multi
plt.figure(figsize=(10, 5))
plt.scatter(range(len(y_test_m)), y_test_m, color='blue', label='Actual')
plt.scatter(range(len(y_test_m)), y_pred_m, color='green', label='Linear Predicted')
plt.scatter(range(len(y_test_m)), y_pred_poly_m, color='red', label='Poly Predicted', alpha=0.6)
plt.legend()
plt.title("Prediction Comparison")
plt.xlabel("Sample")
plt.ylabel("Score")
plt.show()
