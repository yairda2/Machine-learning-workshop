# Author: Yair Davidof 2024
# Importing necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from ucimlrepo import fetch_ucirepo

# Fetch the dataset from the UCI Machine Learning Repository using a predefined function
wine_quality = fetch_ucirepo(id=186)

# Extract features and target variable from the dataset
X = wine_quality.data.features  # Features matrix containing the input variables
y = wine_quality.data.targets.copy()  # Create a copy to avoid SettingWithCopyWarning when modifying

# Convert the numeric quality scores into categorical classes suitable for classification
# Binning quality scores into three categories: low (0), medium (1), high (2)
y.loc[:, 'quality'] = pd.cut(y['quality'], bins=[0, 5, 7, 10], labels=[0, 1, 2])

# Display the structure and summary statistics of the features dataset
print(X.info())
print(X.describe())

# Visualize the distribution of the wine quality categories
y['quality'].value_counts().plot(kind='bar')
plt.title('Distribution of Wine Quality Categories')
plt.xlabel('Quality Category')
plt.ylabel('Frequency')
plt.show()

# Normalize the feature data to have mean=0 and variance=1 to improve model performance
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the normalized data into training and testing sets for model evaluation
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Initialize a Gradient Boosting Classifier.
# This model will be used to classify wine quality
gb_model = GradientBoostingClassifier(random_state=42)

# Set up GridSearchCV to find the best parameters for the model, enhancing prediction accuracy
param_grid = {
    'n_estimators': [100, 200],  # Specifies the number of trees in the model
    'learning_rate': [0.01, 0.1],  # Controls the rate at which the model learns
    'max_depth': [3, 5]  # Limits the number of nodes in each tree
}
# Perform grid search with 10-fold cross-validation
grid_search = GridSearchCV(gb_model, param_grid, cv=10, scoring='accuracy')
grid_search.fit(X_train, y_train.values.ravel())

# Print the best parameters and the best cross-validation score obtained
print("Best parameters:", grid_search.best_params_)
print("Best cross-validation score: {:.2f}".format(grid_search.best_score_))

# Evaluate the best model from grid search on the testing data set
optimal_gb_model = grid_search.best_estimator_
y_pred = optimal_gb_model.predict(X_test)
print("Test set accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Conduct final evaluation using 10-fold cross-validation with the optimal parameters
final_scores = cross_val_score(optimal_gb_model, X_scaled, y.values.ravel(), cv=10, scoring='accuracy')
print("Final CV Accuracy:", np.mean(final_scores))
