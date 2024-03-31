import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.decomposition import TruncatedSVD  # Using TruncatedSVD for sparse data

# Load the dataset
data = pd.read_csv(r"C:\Users\yair\Documents\GitHub\Machine-learning-workshop\AI_survey\ex1\Drug_overdose_death_rates__by_drug_type__sex__age__race__and_Hispanic_origin__United_States (1).csv")
# Define the ColumnTransformer to apply different preprocessing to specified columns
preprocessor = ColumnTransformer(
    transformers=[
        # Apply OneHotEncoder to categorical columns. Adjust 'YEAR' and 'AGE' with actual categorical column names from your dataset.
        ('cat', OneHotEncoder(), ['YEAR', 'AGE']),

        # Apply StandardScaler to numerical columns. Replace 'ESTIMATE' with the actual numerical column names from your dataset.
        ('num', StandardScaler(), ['ESTIMATE'])
    ],
    remainder='drop'  # Drops all other columns not specified
)

# Apply the transformations
try:
    data_transformed = preprocessor.fit_transform(data)
    print("Data successfully transformed.")
except Exception as e:
    print(f"An error occurred during transformation: {e}")

imputer = SimpleImputer(strategy='mean')
data_imputed = imputer.fit_transform(data_transformed)

# Use TruncatedSVD for dimensionality reduction on potentially sparse data
svd = TruncatedSVD(n_components=2)
X_svd = svd.fit_transform(data_imputed)

# Convert the dimensionality-reduced data into a DataFrame for further analysis or visualization
X_svd_df = pd.DataFrame(X_svd, columns=['Component 1', 'Component 2'])

# Display the first few rows to verify
print(X_svd_df.head())


import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))
plt.scatter(X_svd_df['Component 1'], X_svd_df['Component 2'])
plt.xlabel('Component 1')
plt.ylabel('Component 2')
plt.title('2D Projection of Data')
plt.show()
