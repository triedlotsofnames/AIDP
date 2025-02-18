import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Step 1: Load the Preprocessed Dataset
preprocessed_data = "Data/preprocessed.csv"
print(f"Using {preprocessed_data} for model training.")

df = pd.read_csv(preprocessed_data)

# Ensure feature consistency
if 'Survived' not in df.columns:
    raise ValueError("Survived column missing in preprocessed data")

# Drop 'Age_Group' if it exists (since it's categorical and may cause issues)
if 'Age_Group' in df.columns:
    df = df.drop(columns=['Age_Group'])

X = df.drop(columns=["Survived"])
y = df["Survived"]

# Step 2: Split the Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Step 3: Initialize Models
models = {
    "Logistic Regression": LogisticRegression(random_state=42, max_iter=500),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
    "Support Vector Machine": SVC(random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42)
}

# Step 4: Train and Evaluate Models
model_accuracies = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    model_accuracies[name] = accuracy
    print(f"{name} Accuracy: {accuracy:.4f}")

# Step 5: Select the Best Model
best_model_name = max(model_accuracies, key=model_accuracies.get)
best_model = models[best_model_name]
print(f"Best Model: {best_model_name} with Accuracy: {model_accuracies[best_model_name]:.4f}")

# Step 6: Hyperparameter Tuning for the Best Model
param_grids = {
    "Logistic Regression": {
        'C': [0.01, 0.1, 1, 10],
        'solver': ['lbfgs', 'liblinear']
    },
    "Decision Tree": {
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    },
    "Random Forest": {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    },
    "K-Nearest Neighbors": {
        'n_neighbors': [3, 5, 7, 9]
    },
    "Support Vector Machine": {
        'C': [0.1, 1, 10],
        'kernel': ['linear', 'rbf']
    },
    "Gradient Boosting": {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 10]
    }
}

if best_model_name in param_grids:
    param_grid = param_grids[best_model_name]
    grid_search = GridSearchCV(best_model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    print("Best Hyperparameters:", grid_search.best_params_)

# Step 7: Evaluate Best Model
y_pred = best_model.predict(X_test)
print("Test Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Step 8: Save the Best Model for Inference
model_path = "best_titanic_model.pkl"
joblib.dump(best_model, model_path)
print(f"Best model saved as {model_path}")
