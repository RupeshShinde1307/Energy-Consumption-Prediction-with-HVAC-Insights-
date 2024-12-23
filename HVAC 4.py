import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load your dataset (ensure you have hvac_energy.csv in the same directory)
df = pd.read_csv('hvac_energy.csv')

# Define features (independent variables) and target (dependent variable)
X = df[['temperature', 'humidity', 'time_of_day']]
y = df['energy_usage']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Function to get user input and predict energy usage
def predict_energy_usage():
    while True:
        try:
            # Get user input for prediction
            temperature = float(input("Enter temperature (or type 'exit' to quit): "))
            humidity = float(input("Enter humidity (or type 'exit' to quit): "))
            time_of_day = float(input("Enter time of day (or type 'exit' to quit): "))

            # Create DataFrame for the input data
            input_data = pd.DataFrame([[temperature, humidity, time_of_day]], columns=['temperature', 'humidity', 'time_of_day'])
            predicted_energy = model.predict(input_data)

            # Print the predicted energy usage
            print(f"Predicted Energy Usage: {predicted_energy[0]:.2f} units")

        except ValueError:
            print("Invalid input, please enter numeric values.")
        
        except KeyboardInterrupt:
            print("\nExiting the program.")
            break

# Call the function to start the prediction
predict_energy_usage()
