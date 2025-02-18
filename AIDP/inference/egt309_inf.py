import os
import pandas as pd
import joblib

# Paths to data and model
model_path = "data/best_titanic_model.pkl"
test_data_path = "data/preprocessed_test.csv"

# Check if files exist
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found: {model_path}")
if not os.path.exists(test_data_path):
    raise FileNotFoundError(f"Test data file not found: {test_data_path}")

"""**Loading Data**"""
# Load the test dataset
test_data = pd.read_csv(test_data_path)

# Load the saved model
loaded_model = joblib.load(model_path)

# Ensure the test data has the same features as the model expects
expected_features = loaded_model.feature_names_in_  # Works with sklearn models
test_data = test_data[expected_features]  # Select only required columns

"""**Predictions**"""
# Make predictions
predictions = loaded_model.predict(test_data)

# Add predictions to the dataframe
test_data["Survival_Prediction"] = predictions

# Calculate survival percentages
survival_counts = test_data["Survival_Prediction"].value_counts(normalize=True) * 100

# Display results
print(f"Survived: {survival_counts.get(1, 0):.2f}%")
print(f"Not Survived: {survival_counts.get(0, 0):.2f}%")

# Save predictions to a new CSV file
output_path = "data/predictions.csv"
test_data.to_csv(output_path, index=False)
print(f"Predictions saved to {output_path}")

