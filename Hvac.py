import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


# Load dataset
data = pd.read_csv("hvac_energy.csv")

# Display the first few rows of the dataset
print(data.head())

sns.pairplot(data)
plt.show()

# Correlation matrix
print(data.corr())


# Features: temperature, humidity, time_of_day
X = data[['temperature', 'humidity', 'time_of_day']]

# Target: energy usage
y = data['energy_usage']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Create the Linear Regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)


# Calculate MSE
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# Display the model coefficients
print("Model Coefficients:", model.coef_)


plt.scatter(y_test, y_pred)
plt.xlabel('Actual Energy Usage')
plt.ylabel('Predicted Energy Usage')
plt.title('Actual vs Predicted Energy Usage')
plt.show()
