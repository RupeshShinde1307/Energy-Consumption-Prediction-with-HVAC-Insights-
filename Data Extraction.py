import pandas as pd

# Load the dataset
file_path = ("C:/Users/Rupesh Shinde/Desktop/HVAC/hvac_energy.csv")# Adjust the path as needed
data = pd.read_csv(file_path)

# Display the first few rows to check the current format
print("Before conversion:")
print(data.head())

# Convert the 'time_of_day' column (or replace with your column name) to datetime format
# Assume 'time_of_day' is the column containing date and time in string format
data['time_of_day'] = pd.to_datetime(data['time_of_day'], errors='coerce')

# Display the first few rows after conversion to check the new format
print("\nAfter conversion:")
print(data.head())

# Optional: Save the cleaned data back to a new CSV file
data.to_csv("C:/Users/Rupesh Shinde/Desktop/HVAC/hvac_energy_cleaned.csv", index=False)
