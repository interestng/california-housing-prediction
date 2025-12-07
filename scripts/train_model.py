import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import pickle
import os

# load data
df = pd.read_csv('data/housing_processed.csv')

X = df.drop('median_house_value', axis=1)
y = df['median_house_value']

# train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Train: {len(X_train)}, Test: {len(X_test)}")

# train models
print("Training Linear Regression...")
lr = LinearRegression()
lr.fit(X_train, y_train)

print("Training Random Forest...")
rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
rf.fit(X_train, y_train)

# save models
os.makedirs('outputs', exist_ok=True)
pickle.dump(lr, open('outputs/model_lr.pkl', 'wb'))
pickle.dump(rf, open('outputs/model_rf.pkl', 'wb'))

# save test set
X_test.to_csv('data/X_test.csv', index=False)
y_test.to_csv('data/y_test.csv', index=False)

print("Models saved")

