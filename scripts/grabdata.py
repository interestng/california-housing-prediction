from sklearn.datasets import fetch_california_housing
import pandas as pd
import os

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_folder = os.path.join(project_root, "data")

os.makedirs(data_folder, exist_ok=True)

data = fetch_california_housing(as_frame=True)
df = data.frame.rename(columns={'MedHouseVal': 'median_house_value'})

df.to_csv(os.path.join(data_folder, "housing.csv"), index=False)

print("Saved")
