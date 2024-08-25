# Predictive Model Analysis for Vehicle Maintenance

## Overview

This project aims to build and analyze predictive models for vehicle maintenance using a dataset containing vehicle-related features. The analysis includes exploratory data analysis (EDA), feature selection, model evaluation, and comparison of different classifiers. The final model will be used to predict whether a vehicle needs maintenance based on various features.

## Project Structure

1. **Data Loading and EDA**
2. **Data Processing and Feature Engineering**
3. **Model Training and Evaluation**
4. **Model Comparison and Visualization**
5. **Final Model and Prediction**

## Requirements

- Python 3.x
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Matplotlib

## Getting Started

1. **Install Dependencies**

   You can install the necessary libraries using pip:
   ```bash
   pip install pandas numpy scikit-learn xgboost matplotlib
   ```

2. **Dataset**

   Ensure you have the dataset `vehicle_maintenance_data_v2.csv` in the correct path (`/content/vehicle_maintenance_data_v2.csv`).

3. **Run the Script**

   Execute the provided script to perform the following steps:

   - Load and examine the dataset.
   - Process and transform features.
   - Apply statistical tests for feature selection.
   - Train various classifiers and evaluate their performance.
   - Plot performance metrics to compare models.
   - Determine the best model and make predictions.

## Script Walkthrough

### 1. Data Loading and EDA

The script starts by loading the dataset and performing exploratory data analysis (EDA) to understand its structure and basic statistics.

```python
import pandas as pd

df = pd.read_csv('/content/vehicle_maintenance_data_v2.csv')
df.info()
df.shape
df.head()
```

### 2. Data Processing and Feature Engineering

The script processes the dataset by handling categorical features and applying feature selection techniques. The features are encoded, and statistical tests (ANOVA and Chi-Square) are used to assess their relevance.

```python
from sklearn.feature_selection import chi2, f_classif
from sklearn.preprocessing import LabelEncoder

# Convert categorical features to numeric
label_encoders = {}
for column in X.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    X[column] = le.fit_transform(X[column])
    label_encoders[column] = le

# Apply ANOVA for numerical features
X_numeric = X.select_dtypes(include=['int64', 'float64'])
f_values, p_values = f_classif(X_numeric, y)

# Apply Chi-Square for categorical features
X_categorical = X[[col for col in X]]  # Select columns based on label_encoders dictionary
chi2_values, p_values = chi2(X_categorical, y)
```

### 3. Model Training and Evaluation

Different classifiers are trained and evaluated using Recursive Feature Elimination (RFE) to determine the optimal number of features.

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Split and scale data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train and evaluate models
models = {
    'SVC': SVC(kernel='linear', probability=True),
    'XGBoost': XGBClassifier(),
    'Gradient Boosting': GradientBoostingClassifier(),
    'Random Forest': RandomForestClassifier()
}

results = {model_name: {'features': [], 'accuracy': [], 'precision': [], 'recall': [], 'f1_score': [], 'roc_auc': []} for model_name in models}

# Evaluate models with RFE
for model_name, model in models.items():
    for n_features_to_select in range(3, 11):  # Try different numbers of features from 3 to 10
        rfe = RFE(estimator=model, n_features_to_select=n_features_to_select)
        X_train_rfe = rfe.fit_transform(X_train_scaled, y_train)
        X_test_rfe = rfe.transform(X_test_scaled)
        model.fit(X_train_rfe, y_train)
        y_pred = model.predict(X_test_rfe)
        y_pred_proba = model.predict_proba(X_test_rfe)[:, 1]
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        results[model_name]['features'].append(n_features_to_select)
        results[model_name]['accuracy'].append(accuracy)
        results[model_name]['precision'].append(precision)
        results[model_name]['recall'].append(recall)
        results[model_name]['f1_score'].append(f1)
        results[model_name]['roc_auc'].append(roc_auc)
```

### 4. Model Comparison and Visualization

Performance metrics are plotted to compare models and determine the best one based on accuracy and other metrics.

```python
import matplotlib.pyplot as plt

# Plot performance metrics
plt.figure(figsize=(20, 15))
for i, (model_name, metrics) in enumerate(results.items()):
    plt.subplot(2, 2, i+1)
    plt.plot(metrics['features'], metrics['accuracy'], marker='o', label='Accuracy')
    plt.plot(metrics['features'], metrics['precision'], marker='o', label='Precision')
    plt.plot(metrics['features'], metrics['recall'], marker='o', label='Recall')
    plt.plot(metrics['features'], metrics['f1_score'], marker='o', label='F1-Score')
    plt.plot(metrics['features'], metrics['roc_auc'], marker='o', label='ROC AUC')
    plt.title(f'{model_name} Performance vs Number of Features')
    plt.xlabel('Number of Features')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True)

plt.tight_layout()
plt.show()
```

### 5. Final Model and Prediction

The final model is selected based on performance metrics, retrained on the optimal features, and used to make predictions on new data.

```python
# Select the best features and train the final model
features = ['Mileage', 'Maintenance_History', 'Vehicle_Age', 'Fuel_Type', 'Odometer_Reading', 'Tire_Condition', 'Battery_Status']
X_selected = X[features]
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
model = XGBClassifier()
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)

# Evaluate the final model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1-Score: {f1:.4f}')

# Predict on new data
new_data = pd.DataFrame({
    'Mileage': [15000],
    'Maintenance_History': [1],
    'Vehicle_Age': [5],
    'Fuel_Type': [1],
    'Odometer_Reading': [60000],
    'Tire_Condition': [2],
    'Battery_Status': [0]
})
new_data_scaled = scaler.transform(new_data)
prediction = model.predict(new_data_scaled)
print(f'Prediction: {prediction}')
```

## Conclusion

The script provides a comprehensive analysis of vehicle maintenance data, evaluates multiple models, and identifies the best-performing model for predicting maintenance needs. The final model's predictions can be used to inform maintenance schedules and optimize vehicle management.

Feel free to modify and extend the analysis to fit specific needs or incorporate additional data features.
