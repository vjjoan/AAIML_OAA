import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score

# Improved Simulated Dataset (more distinct characteristics)
np.random.seed(42)

appliance_names = ['Fridge', 'Microwave', 'TV', 'Washing Machine', 'AC']
n_samples = 300  # Increased number of samples
data = {
    'mean_power': np.concatenate([
        np.random.normal(150, 30, 60),     # Fridge (higher base power)
        np.random.normal(800, 150, 60),    # Microwave (high power, short duration)
        np.random.normal(100, 20, 60),     # TV (lower power)
        np.random.normal(300, 70, 60),     # Washing Machine (moderate power, longer duration cycles)
        np.random.normal(1000, 200, 60)    # AC (high power, longer duration)
    ]),
    'max_power': np.concatenate([
        np.random.normal(200, 50, 60),
        np.random.normal(1200, 200, 60),
        np.random.normal(150, 30, 60),
        np.random.normal(500, 100, 60),
        np.random.normal(1500, 300, 60)
    ]),
    'std_dev_power': np.concatenate([
        np.random.normal(15, 5, 60),      # Fridge (stable power)
        np.random.normal(100, 30, 60),     # Microwave (fluctuations)
        np.random.normal(10, 3, 60),      # TV (stable)
        np.random.normal(80, 20, 60),      # Washing Machine (cycles)
        np.random.normal(150, 40, 60)     # AC (fluctuations with compressor)
    ]),
    'duration': np.concatenate([
        np.random.uniform(60, 180, 60),   # Fridge (long, continuous)
        np.random.uniform(1, 15, 60),     # Microwave (short)
        np.random.uniform(30, 120, 60),   # TV (moderate)
        np.random.uniform(30, 90, 60),    # Washing Machine (moderate to long cycles)
        np.random.uniform(60, 240, 60)    # AC (long)
    ]),
    'appliance': np.concatenate([np.repeat(appliance_names[i], 60) for i in range(len(appliance_names))])
}

df = pd.DataFrame(data)

# Encode target labels
df['appliance_code'] = df['appliance'].astype('category').cat.codes
appliance_map = dict(enumerate(df['appliance'].astype('category').cat.categories))

# Features and target
X = df[['mean_power', 'max_power', 'std_dev_power', 'duration']]
y = df['appliance_code']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# --- Random Forest with Hyperparameter Tuning ---
param_grid_rf = {
    'n_estimators': [50, 100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'class_weight': ['balanced', None]
}

grid_search_rf = GridSearchCV(RandomForestClassifier(random_state=42), param_grid_rf, cv=3, scoring='accuracy', n_jobs=-1) # n_jobs=-1 uses all available cores
grid_search_rf.fit(X_train, y_train)

# Get the Random Forest model
best_rf_clf = grid_search_rf.best_estimator_

# Prediction
y_pred_rf = best_rf_clf.predict(X_test)

# Evaluation
accuracy_rf = accuracy_score(y_test, y_pred_rf)
report_rf = classification_report(y_test, y_pred_rf, target_names=appliance_names)

print("--- Random Forest Model ---")
print("Hyperparameters:", grid_search_rf.best_params_)
print("Accuracy:", accuracy_rf)
print("Classification Report:\n", report_rf)

# Map and display sample predictions
print("\nSample Predictions:")
for i in range(10):
    actual = appliance_map[y_test.iloc[i]]
    predicted = appliance_map[y_pred_rf[i]]
    print(f"Predicted: {predicted}, Actual: {actual}")
