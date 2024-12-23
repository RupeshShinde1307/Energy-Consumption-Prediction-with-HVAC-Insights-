import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv(r'C:/Users/Rupesh Shinde/Desktop/HVAC/Energy_consumption.csv')

# Convert 'time_of_day' to datetime and extract hour
df['time_of_day'] = pd.to_datetime(df['time_of_day'])
df['hour'] = df['time_of_day'].dt.hour

# Convert categorical fields to numerical data using label encoding
le = LabelEncoder()
df['HVACUsage'] = le.fit_transform(df['HVACUsage'])  # 'On' -> 1, 'Off' -> 0
df['LightingUsage'] = le.fit_transform(df['LightingUsage'])  # 'On' -> 1, 'Off' -> 0
df['DayOfWeek'] = le.fit_transform(df['DayOfWeek'])  # 'Monday' -> 0, 'Sunday' -> 6
df['Holiday'] = le.fit_transform(df['Holiday'])  # 'Yes' -> 1, 'No' -> 0

# Define features and target
X = df[['temperature', 'humidity', 'SquareFootage', 'Occupancy', 'HVACUsage', 'LightingUsage', 
        'RenewableEnergy', 'DayOfWeek', 'Holiday', 'hour']]
y = df['energy_usage']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)

# Function to predict and plot energy usage
def predict_and_plot_energy_usage():
    while True:
        try:
            # Get user input for all required fields
            temp = float(input("Enter temperature: "))
            humidity = float(input("Enter humidity: "))
            square_footage = float(input("Enter SquareFootage: "))
            occupancy = int(input("Enter number of occupants: "))
            hvac_usage = int(input("Is HVAC on? (1 for Yes, 0 for No): "))
            lighting_usage = int(input("Is Lighting on? (1 for Yes, 0 for No): "))
            renewable_energy = float(input("Enter renewable energy used: "))
            day_of_week = int(input("Enter day of week (0 for Monday, 6 for Sunday): "))
            holiday = int(input("Is it a holiday? (1 for Yes, 0 for No): "))
            hour = int(input("Enter hour of the day (0-23): "))

            # Create a DataFrame for the input
            input_data = pd.DataFrame([[temp, humidity, square_footage, occupancy, hvac_usage, 
                                        lighting_usage, renewable_energy, day_of_week, holiday, hour]], 
                                      columns=['temperature', 'humidity', 'SquareFootage', 'Occupancy', 
                                               'HVACUsage', 'LightingUsage', 'RenewableEnergy', 
                                               'DayOfWeek', 'Holiday', 'hour'])

            # Predict energy usage based on input
            predicted_energy = model.predict(input_data)
            print(f"Predicted energy usage: {predicted_energy[0]:.2f} kWh")

            # Scatter plot: Actual vs Predicted Energy Usage
            plt.figure(figsize=(10, 6))
            plt.scatter(y_test, y_pred, color='blue', alpha=0.5, label='Predicted vs Actual')
            plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linewidth=2, label='Ideal Prediction')
            plt.xlabel('Actual Energy Usage')
            plt.ylabel('Predicted Energy Usage')
            plt.title('Actual vs Predicted Energy Usage')
            plt.legend()
            plt.grid(True)
            plt.show()

            # Line plot: Energy usage over time (predicted values)
            plt.figure(figsize=(10, 6))
            plt.plot(range(len(y_test)), y_test, color='green', label='Actual Energy Usage')
            plt.plot(range(len(y_pred)), y_pred, color='blue', linestyle='--', label='Predicted Energy Usage')
            plt.xlabel('Sample Index')
            plt.ylabel('Energy Usage (kWh)')
            plt.title('Energy Usage Over Time')
            plt.legend()
            plt.grid(True)
            plt.show()

            # Histogram of predicted energy usage
            plt.figure(figsize=(10, 6))
            plt.hist(y_pred, bins=20, color='orange', alpha=0.7)
            plt.title('Distribution of Predicted Energy Usage')
            plt.xlabel('Energy Usage (kWh)')
            plt.ylabel('Frequency')
            plt.grid(True)
            plt.show()

        except ValueError:
            print("Please enter valid numerical values.")
        except KeyboardInterrupt:
            print("Exiting...")
            break

# Call the prediction and plotting function
predict_and_plot_energy_usage()
