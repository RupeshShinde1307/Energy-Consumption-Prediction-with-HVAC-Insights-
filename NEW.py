import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load your data
data = pd.read_csv('C:\\Users\\Rupesh Shinde\\Desktop\\HVAC\\hvac_energy.csv')

# Assuming 'datetime_column' is the name of your column with date-time values
data['datetime_column'] = pd.to_datetime(data['datetime_column'], errors='coerce')

# Extract date and time features
data['year'] = data['datetime_column'].dt.year
data['month'] = data['datetime_column'].dt.month
data['day'] = data['datetime_column'].dt.day
data['hour'] = data['datetime_column'].dt.hour

# Drop the original datetime column
data = data.drop(columns=['datetime_column'])

# Separate features and target variable
X = data.drop(columns=['target_column'])  # Replace 'target_column' with the actual column name
y = data['target_column']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict and evaluate (optional)
predictions = model.predict(X_test)
