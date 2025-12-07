import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# load models
lr = pickle.load(open('outputs/model_lr.pkl', 'rb'))
rf = pickle.load(open('outputs/model_rf.pkl', 'rb'))

# load test data
X_test = pd.read_csv('data/X_test.csv')
y_test = pd.read_csv('data/y_test.csv').values.ravel()

# make predictions
pred_lr = lr.predict(X_test)
pred_rf = rf.predict(X_test)

# calculate metrics for linear regression
rmse_lr = np.sqrt(mean_squared_error(y_test, pred_lr))
mae_lr = mean_absolute_error(y_test, pred_lr)
r2_lr = r2_score(y_test, pred_lr)

print("Linear Regression:")
print(f"  RMSE: {rmse_lr:.2f}")
print(f"  MAE: {mae_lr:.2f}")
print(f"  R²: {r2_lr:.4f}")

# calculate metrics for random forest
rmse_rf = np.sqrt(mean_squared_error(y_test, pred_rf))
mae_rf = mean_absolute_error(y_test, pred_rf)
r2_rf = r2_score(y_test, pred_rf)

print("\nRandom Forest:")
print(f"  RMSE: {rmse_rf:.2f}")
print(f"  MAE: {mae_rf:.2f}")
print(f"  R²: {r2_rf:.4f}")

# plot predictions
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.scatter(y_test, pred_lr, alpha=0.5)
ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
ax1.set_xlabel('Actual')
ax1.set_ylabel('Predicted')
ax1.set_title('Linear Regression')
ax1.grid(True, alpha=0.3)

ax2.scatter(y_test, pred_rf, alpha=0.5)
ax2.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
ax2.set_xlabel('Actual')
ax2.set_ylabel('Predicted')
ax2.set_title('Random Forest')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
os.makedirs('outputs/plots', exist_ok=True)
plt.savefig('outputs/plots/predicted_vs_actual.png', dpi=300, bbox_inches='tight')
print("\nSaved predicted vs actual plot")

# feature importance
importance_df = pd.DataFrame({
    'feature': X_test.columns,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(data=importance_df, x='importance', y='feature')
plt.title('Random Forest Feature Importance')
plt.tight_layout()
plt.savefig('outputs/plots/feature_importance.png', dpi=300, bbox_inches='tight')
print("Saved feature importance plot")

