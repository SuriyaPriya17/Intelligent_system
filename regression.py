import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

data = fetch_california_housing()
X, y = data.data, data.target
feature_names = data.feature_names
X_simple = X[:, [0]]

X_train, X_test, y_train, y_test = train_test_split(X_simple, y, test_size=0.2, random_state=42)

simple_model = LinearRegression()
simple_model.fit(X_train, y_train)
y_pred_simple = simple_model.predict(X_test)

print("\nSimple Linear Regression (MedInc → Price)")
print("MAE:", mean_absolute_error(y_test, y_pred_simple))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_simple)))
print("R² Score:", r2_score(y_test, y_pred_simple))

plt.scatter(X_test, y_test, color="blue", alpha=0.3, label="Actual")
plt.plot(X_test, y_pred_simple, color="red", linewidth=2, label="Predicted Line")
plt.xlabel("Median Income")
plt.ylabel("Median House Value (in $100,000s)")
plt.title("Simple Linear Regression - California Housing")
plt.legend()
plt.show()


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

multi_model = LinearRegression()
multi_model.fit(X_train, y_train)
y_pred_multi = multi_model.predict(X_test)

print("\nMultiple Linear Regression (All Features → Price)")
print("MAE:", mean_absolute_error(y_test, y_pred_multi))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_multi)))
print("R² Score:", r2_score(y_test, y_pred_multi))

print("\nCoefficients per feature:")
for name, coef in zip(feature_names, multi_model.coef_):
    print(f"{name}: {coef:.4f}")
print("Intercept:", multi_model.intercept_)
