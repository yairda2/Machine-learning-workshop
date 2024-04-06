# Author: Yair Davidof. 2024
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd  # Import pandas for additional handling of the dataset
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from ucimlrepo import fetch_ucirepo

# Fetch the dataset
wine_quality = fetch_ucirepo(id=186)

# Data (as pandas DataFrames)
X = wine_quality.data.features  # Features
y = wine_quality.data.targets  # Target variable

# Convert the numeric quality scores into categorical classes
# Define bins for quality scores to categorize wine quality:
# Assuming 'quality' is a numeric score, categorize them into low, medium, high
y['quality'] = pd.cut(y['quality'], bins=[0, 5, 7, 10], labels=[0, 1, 2])

# Basic information about features
print(X.info())

# Descriptive statistics of the features
print(X.describe())

# Distribution of the target variable 'quality' after categorization
y['quality'].value_counts().plot(kind='bar')
plt.title('Distribution of Wine Quality Categories')
plt.xlabel('Quality Category')
plt.ylabel('Frequency')
plt.show()

# Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Splitting data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Initialize and train the GradientBoostingClassifier
model = GradientBoostingClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train.values.ravel())  # Fit model on training data

# Predict on the test set
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Feature Importance
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]

# Plotting feature importances
plt.figure(figsize=(10, 6))
plt.title('Feature Importance')
plt.bar(range(X.shape[1]), importances[indices])
plt.xticks(range(X.shape[1]), X.columns[indices], rotation=90)
plt.tight_layout()
plt.show()

# Cross-validation to evaluate model
scores = cross_val_score(model, X_scaled, y.values.ravel(), cv=10, scoring='accuracy')
print("CV Accuracy:", np.mean(scores))
