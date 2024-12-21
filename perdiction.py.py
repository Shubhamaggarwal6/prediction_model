import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

# Load the dataset
file_path = 'Fuel_cell_performance_data-Full.csv'  # Replace with your file path
fuel_cell_data = pd.read_csv(file_path)

# Get the roll number (interactive or via command-line argument)
if len(sys.argv) > 1:
    roll_number = int(sys.argv[1])
else:
    roll_number = int(input("Enter your roll number: "))

# Extract the last digit of the roll number
roll_number_ending = roll_number % 10

# Determine the target column based on the last digit of the roll number
if roll_number_ending in [0, 5]:
    target_column = 'Target1'
elif roll_number_ending in [1, 6]:
    target_column = 'Target2'
elif roll_number_ending in [2, 7]:
    target_column = 'Target3'
elif roll_number_ending in [3, 8]:
    target_column = 'Target4'
elif roll_number_ending in [4, 9]:
    target_column = 'Target5'
else:
    raise ValueError("Invalid roll number ending!")

# Drop other target columns and prepare features and target
X = fuel_cell_data[['F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'F9', 'F10', 'F11', 'F12', 'F13', 'F14', 'F15']]
y = fuel_cell_data[target_column]

# Split the dataset into 70% training and 30% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize models
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42)
}

# Train and evaluate models
results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    results[name] = mse
    print(f"{name} - Mean Squared Error: {mse:.4f}")

# Display the best model
best_model = min(results, key=results.get)
print(f"Best model: {best_model} with MSE: {results[best_model]:.4f}")

# Add the roll number to the results and save to CSV
results_df = pd.DataFrame(list(results.items()), columns=["Model", "Mean Squared Error"])
results_df["Roll Number"] = roll_number  # Add the roll number column

# Save results to a CSV file
results_df.to_csv('results.csv', index=False)
print("Results have been saved to 'results.csv'")
