import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
import numpy as np

# --- 1. Load the Data ---
try:
    df = pd.read_csv('train.csv')
except FileNotFoundError:
    print("Error: 'train.csv' not found. Please make sure the file is in the same directory.")
    exit()

# --- 2. Data Preprocessing ---
print("--- Starting Data Preprocessing ---")

# Handle missing 'loan_int_rate' with the median (Updated to avoid FutureWarning)
median_int_rate = df['loan_int_rate'].median()
df['loan_int_rate'] = df['loan_int_rate'].fillna(median_int_rate)
print(f"Filled missing 'loan_int_rate' with median: {median_int_rate}")

# Handle missing 'person_emp_length' with the median (Updated to avoid FutureWarning)
median_emp_length = df['person_emp_length'].median()
df['person_emp_length'] = df['person_emp_length'].fillna(median_emp_length)
print(f"Filled missing 'person_emp_length' with median: {median_emp_length}")


# Convert categorical variables to numerical using one-hot encoding
# This creates new columns for each category (e.g., 'person_home_ownership_RENT')
df = pd.get_dummies(df, columns=['person_home_ownership', 'loan_intent', 'loan_grade', 'cb_person_default_on_file'], drop_first=True)
print("Converted categorical features to numerical format.")

# Log transform skewed numerical features to normalize their distribution
df['person_income_log'] = np.log1p(df['person_income'])
df['loan_amnt_log'] = np.log1p(df['loan_amnt'])
print("Applied log transformation to 'person_income' and 'loan_amnt'.")

# Drop the original columns and the ID, which is not a feature
df = df.drop(['id', 'person_income', 'loan_amnt'], axis=1)

# --- 3. Prepare Data for Modeling ---
# Separate features (X) from the target variable (y)
X = df.drop('loan_status', axis=1)
y = df['loan_status']

# Split the data into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print("\n--- Data split into training and testing sets ---")

# Scale numerical features
# This ensures all features have a similar scale, which is important for Logistic Regression
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
print("Scaled numerical features using StandardScaler.")

# --- 4. Build and Train the Baseline Model ---
print("\n--- Training the Logistic Regression Model ---")
# Initialize the model
baseline_model = LogisticRegression(random_state=42)

# Train the model on the training data
baseline_model.fit(X_train, y_train)
print("Model training complete.")

# --- 5. Evaluate the Model ---
print("\n--- Evaluating Baseline Model Performance ---")
# Make predictions on the test data
y_pred = baseline_model.predict(X_test)

# Calculate and print the accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Print the Confusion Matrix
# This shows how many predictions were correct/incorrect for each class
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Print the Classification Report
# This gives detailed metrics like precision, recall, and f1-score for each class
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Not Defaulted (0)', 'Defaulted (1)']))

print("\n--- Script Finished ---")

