import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline

# Example DataFrame setup (replace this with your actual DataFrame)
df = pd.read_csv('percentage_changes_decr_price/2012_percentage_changes.csv')

df = df[(df['dept_id'] == 'FOODS_1') & (df['store_id'] == 'TX_1')]
df['price_change'] = df['price_change'].abs()
df = df.dropna()

# Extract features (price_change) and target variable (sales_change)
X = df[['price_change']].values
X = np.clip(X, -100, 100)
y = df['sales_change'].values
y = np.clip(y, -100, 2000)

# Define polynomial degree
degree = 3  # You can change this to any degree you want

# Create polynomial features
poly_features = PolynomialFeatures(degree=degree, include_bias=False)
X_poly = poly_features.fit_transform(X.reshape(-1, 1))

# Create polynomial regression model
model = LinearRegression()

# Fit the model
model.fit(X_poly, y)

# Assuming X_new contains new data points for price_change
X_new = np.arange(0,101).reshape(-1, 1)  # Example new data
# Predict sales change

X_new_poly = poly_features.transform(X_new)
# Transform new data using polynomial features

y_pred = model.predict(X_new_poly)

# Plot actual vs predicted sales change
plt.scatter(X, y, color='blue', label='Actual Sales Change')
plt.plot(X_new, y_pred, color='red', label='Predicted Sales Change')
plt.xlabel('Price Change')
plt.ylabel('Sales Change')
dep = df['dept_id'].unique()
plt.title(f'Price Elasticity for {dep}')
plt.legend()
plt.grid(True)

plt.show()
