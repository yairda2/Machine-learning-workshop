# Importing necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA  # For dimensionality reduction
from ucimlrepo import fetch_ucirepo

# Fetch the dataset from the UCI Machine Learning Repository using a predefined function
wine_quality = fetch_ucirepo(id=186)

# Extract features and target variable from the dataset
X = wine_quality.data.features  # Features matrix containing the input variables
y = wine_quality.data.targets.copy()  # Create a copy to avoid SettingWithCopyWarning when modifying

# Convert the numeric quality scores into categorical classes suitable for classification
# Binning quality scores into three categories: low (0), medium (1), high (2)
y.loc[:, 'quality'] = pd.cut(y['quality'], bins=[0, 5, 7, 10], labels=[0, 1, 2])

# Visualize the initial distribution of the wine quality categories
plt.figure(figsize=(6, 4))
y['quality'].value_counts().plot(kind='bar')
plt.title('Initial Distribution of Wine Quality Categories')
plt.xlabel('Quality Category')
plt.ylabel('Frequency')
plt.show()

# Normalize the feature data to have mean=0 and variance=1 to improve model performance
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Using PCA to reduce dimensions for visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Split the normalized and PCA reduced data into training and testing sets for model evaluation
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)

# Plot the PCA-transformed features before classification
plt.figure(figsize=(8, 6))
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train['quality'].astype(int), cmap='viridis', alpha=0.5)
plt.title('PCA of Features by Quality Category Before Classification')
plt.colorbar(label='Quality Category')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()

# Initialize a Gradient Boosting Classifier.
gb_model = GradientBoostingClassifier(random_state=42)

# Set up GridSearchCV to find the best parameters for the model
param_grid = {
    'n_estimators': [100, 200],
    'learning_rate': [0.01, 0.1],
    'max_depth': [3, 5]
}
# Perform grid search with 10-fold cross-validation
grid_search = GridSearchCV(gb_model, param_grid, cv=10, scoring='accuracy')
grid_search.fit(X_train, y_train.values.ravel())

# Evaluate the best model from grid search on the testing data set
optimal_gb_model = grid_search.best_estimator_
y_pred = optimal_gb_model.predict(X_test)

# Visualize the PCA-transformed features after classification
plt.figure(figsize=(8, 6))
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_pred, cmap='viridis', alpha=0.5)
plt.title('PCA of Features by Predicted Quality Category After Classification')
plt.colorbar(label='Predicted Quality Category')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()

# Print accuracy and other metrics
print("Test set accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Conduct final evaluation using 10-fold cross-validation with the optimal parameters
final_scores = cross_val_score(optimal_gb_model, X_scaled, y.values.ravel(), cv=10, scoring='accuracy')
print("Final CV Accuracy:", np.mean(final_scores))

# Display the final results
plt.show()
