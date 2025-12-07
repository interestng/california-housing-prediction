import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pickle
import os

# load dataset
df = pd.read_csv('data/housing.csv')

print(f"Shape: {df.shape}")
print(f"Missing values: {df.isnull().sum().sum()}")

# drop any missing values
df = df.dropna()

# get feature columns (everything except target)
target = 'median_house_value'
feature_cols = [col for col in df.columns if col != target]

# check for categorical columns
cat_cols = []
num_cols = []
for col in feature_cols:
    if df[col].dtype == 'object':
        cat_cols.append(col)
    else:
        num_cols.append(col)

print(f"Numerical features: {len(num_cols)}")
print(f"Categorical features: {len(cat_cols)}")

# encode categoricals if any
encoders = {}
if cat_cols:
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

# scale numerical features
scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

# save processed data
df.to_csv('data/housing_processed.csv', index=False)

# save preprocessing objects
os.makedirs('outputs', exist_ok=True)
pickle.dump(scaler, open('outputs/scaler.pkl', 'wb'))
if encoders:
    pickle.dump(encoders, open('outputs/label_encoders.pkl', 'wb'))

print("Done preprocessing")

