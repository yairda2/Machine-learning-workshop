import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt

# Replace 'your_file_path.csv' with the actual path to your CSV file
data = pd.read_csv(r"C:\Users\yair\Documents\GitHub\Machine-learning-workshop\AI_survey\ex2\Drug_overdose_death_rates__by_drug_type__sex__age__race__and_Hispanic_origin__United_States (1).csv")

if 'DATE_COLUMN' in data.columns:
    data['DATE_COLUMN'] = pd.to_datetime(data['DATE_COLUMN'])
    data['DAY_OF_WEEK'] = data['DATE_COLUMN'].dt.dayofweek
    data['MONTH'] = data['DATE_COLUMN'].dt.month
else:
    print("The date column does not exist in the dataframe.")

# Replace 'CATEGORICAL_COLUMN' with the actual name of the categorical column you want to encode
# Check if 'CATEGORICAL_COLUMN' exists in the dataframe
if 'CATEGORICAL_COLUMN' in data.columns:
    data = pd.get_dummies(data, columns=['CATEGORICAL_COLUMN'])
else:
    print("The categorical column does not exist in the dataframe.")

# Assuming 'ESTIMATE' is a column that contains numeric values to be scaled
# Check if 'ESTIMATE' exists in the dataframe
if 'ESTIMATE' in data.columns:
    # Apply SimpleImputer and StandardScaler to the 'ESTIMATE' column
    imputer = SimpleImputer(strategy='median')
    data['ESTIMATE'] = imputer.fit_transform(data[['ESTIMATE']])
    scaler = StandardScaler()
    data['ESTIMATE'] = scaler.fit_transform(data[['ESTIMATE']])
else:
    print("The numeric column does not exist in the dataframe.")

# Apply TruncatedSVD for dimensionality reduction
svd = TruncatedSVD(n_components=2)
principal_components = svd.fit_transform(data.select_dtypes(include=[np.number]))

# Create a DataFrame for the SVD results
principal_df = pd.DataFrame(data=principal_components, columns=['Principal Component 1', 'Principal Component 2'])

# Visualize the results with a scatter plot
plt.figure(figsize=(8, 6))
plt.scatter(principal_df['Principal Component 1'], principal_df['Principal Component 2'], s=50)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('2D Projection from TruncatedSVD')
plt.show()
