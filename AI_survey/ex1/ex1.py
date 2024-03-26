import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA

# Loading the dataset
record_path = "C:\\Users\\yair\\Documents\\GitHub\\Machine-learning-workshop\\AI_survey\\ex1\\Drug_overdose_death_rates__by_drug_type__sex__age__race__and_Hispanic_origin__United_States (1).csv"
data = pd.read_csv(record_path)

# Drop the 'FLAG' column to avoid issues since it's non-numeric and likely not useful for our analysis
data = data.drop(columns=['FLAG'])

# Drop rows where 'ESTIMATE' is NaN to ensure the target variable contains no missing values
data = data.dropna(subset=['ESTIMATE'])

# Proceed with your preprocessing steps...

# Creating a new column 'Year_Category' based on the 'YEAR'
data['Year_Category'] = data['YEAR'].apply(lambda x: 'Before_2000' if x < 2000 else 'After_2000')

# One-hot Encoding: Transforming categorical variables into a machine learning model-friendly format
# It's important to encode before the imputation step to ensure all data is numeric
data_encoded = pd.get_dummies(data, columns=['PANEL', 'AGE', 'Year_Category'])

# Prepare features (X) and target variable (y) for feature selection and modeling
# Ensure 'X' only includes numeric columns at this point
X = data_encoded.drop(['ESTIMATE', 'INDICATOR', 'UNIT', 'STUB_NAME', 'STUB_LABEL'], axis=1)
y = data_encoded['ESTIMATE']

# Handling missing values in features before feature selection
# The imputation should be applied only on numeric data, which is ensured by the encoding and column filtering above
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

# Feature Selection using SelectKBest
selector = SelectKBest(f_classif, k=10)
X_selected = selector.fit_transform(X_imputed, y)

# Standardization of features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_selected)

# Dimensionality Reduction with PCA
pca = PCA(n_components=5)  # Adjust based on your specific needs
X_pca = pca.fit_transform(X_scaled)

# Now, X_pca is ready for further analysis or modeling
print(X_pca)

# Convert the PCA-transformed numpy array to a DataFrame for easier handling
X_pca_df = pd.DataFrame(X_pca, columns=[f'PC{i+1}' for i in range(X_pca.shape[1])])

# Save the DataFrame to a new CSV file
output_path = "C:\\Users\\yair\\Documents\\GitHub\\Machine-learning-workshop\\AI_survey\\ex1\\PCA_transformed_dataset.csv"
X_pca_df.to_csv(output_path, index=False)

print(f"PCA-transformed dataset saved to {output_path}")

