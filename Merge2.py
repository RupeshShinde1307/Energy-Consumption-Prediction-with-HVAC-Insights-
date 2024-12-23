import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib

# Force backend to TkAgg if it's not already
matplotlib.use('TkAgg')

# Load your dataset
df = pd.read_csv('hvac_energy_cleaned.csv')

# Define features and target
X = df[['temperature', 'humidity', 'time_of_day']]
y = df['energy_usage']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Function to get user input and predict energy usage
def predict_energy_usage():
    while True:
        try:
            # Get user input for prediction
            temperature = input("Enter temperature (or type 'exit' to quit): ")
            if temperature.lower() == 'exit':
                break
            temperature = float(temperature)

            humidity = input("Enter humidity (or type 'exit' to quit): ")
            if humidity.lower() == 'exit':
                break
            humidity = float(humidity)

            time_of_day = input("Enter time of day (or type 'exit' to quit): ")
            if time_of_day.lower() == 'exit':
                break
            time_of_day = float(time_of_day)

            # Create DataFrame for input data
            input_data = pd.DataFrame([[temperature, humidity, time_of_day]], columns=['temperature', 'humidity', 'time_of_day'])
            predicted_energy = model.predict(input_data)

            # Print predicted energy usage
            print(f"Predicted Energy Usage: {predicted_energy[0]:.2f} units")

            # Plot the result
            plot_energy_usage(predicted_energy)

        except ValueError:
            print("Invalid input, please enter numeric values.")
        
        except KeyboardInterrupt:
            print("\nExiting the program.")
            break

# Function to plot the predicted energy usage
def plot_energy_usage(predicted_energy):
    plt.figure(figsize=(8, 5))
    plt.bar(['Predicted Energy'], predicted_energy, color='blue', alpha=0.7)
    plt.ylabel('Predicted Energy Usage (units)')
    plt.title('Predicted Energy Usage based on Input')
    plt.ylim(0, max(predicted_energy[0] + 100, 1000))
    plt.axhline(y=0, color='k', linewidth=1)
    
    # Annotate value on bar
    plt.text(0, predicted_energy[0] + 10, f"{predicted_energy[0]:.2f}", ha='center', fontweight='bold')
    
    plt.grid(axis='y')
    
    # Show plot
    plt.show(block=True)  # Ensure plot stays open until manually closed

# Start the prediction loop
predict_energy_usage()
