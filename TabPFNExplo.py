import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (mean_squared_error, r2_score, mean_absolute_error,
                            explained_variance_score)
from sklearn.inspection import permutation_importance
from tabpfn import TabPFNRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from ioh import problem

# Set plotting style
plt.style.use('ggplot')
sns.set(font_scale=1.2)

# Load and prepare data
print("Loading and preparing data...")
df = fetch_openml(data_id=531, as_frame=True)  # Boston Housing dataset
X = df.data
y = df.target.astype(float)  # Ensure target is float for regression
feature_names = X.columns.tolist()

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=feature_names)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled_df, y, test_size=0.3, random_state=42
)

print(f"Training set: {X_train.shape[0]} samples, Test set: {X_test.shape[0]} samples")
print(f"Number of features: {X_train.shape[1]}")

# Initialize and train TabPFN regressor
print("\nTraining TabPFN regressor...")
tabpfn = TabPFNRegressor(device='cuda')
tabpfn.fit(X_train, y_train)

# Make predictions
y_pred = tabpfn.predict(X_test)

# Compare with traditional models
print("\nComparing with traditional models...")
models = {
    'TabPFN': tabpfn,
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'Ridge': Ridge(alpha=1.0, random_state=42)
}

# Train other models
for name, model in models.items():
    if name != 'TabPFN':  # TabPFN is already trained
        model.fit(X_train, y_train)

# Create visualizations
fig = plt.figure(figsize=(20, 15))
fig.suptitle('TabPFN Regression Analysis - Boston Housing Dataset', fontsize=20)

# 1. Predicted vs Actual Values
ax1 = plt.subplot(2, 2, 1)
ax1.scatter(y_test, y_pred, alpha=0.6)
ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
ax1.set_xlabel('Actual Values')
ax1.set_ylabel('Predicted Values')
ax1.set_title('TabPFN: Predicted vs Actual')

# 2. Residuals Plot
ax2 = plt.subplot(2, 2, 2)
residuals = y_test - y_pred
ax2.scatter(y_pred, residuals, alpha=0.6)
ax2.axhline(y=0, color='r', linestyle='--')
ax2.set_xlabel('Predicted Values')
ax2.set_ylabel('Residuals')
ax2.set_title('Residuals Plot')

# 3. Feature Importance (top 10)
ax3 = plt.subplot(2, 2, 3)
perm_importance = permutation_importance(tabpfn, X_test, y_test, n_repeats=10, random_state=42)
feature_importance = perm_importance.importances_mean
sorted_idx = np.argsort(feature_importance)[-10:]

ax3.barh(range(10), feature_importance[sorted_idx])
ax3.set_yticks(range(10))
ax3.set_yticklabels([feature_names[i] for i in sorted_idx])
ax3.set_title('Top 10 Feature Importance')

# 4. Distribution of Residuals
ax4 = plt.subplot(2, 2, 4)
sns.histplot(residuals, kde=True, ax=ax4)
ax4.axvline(0, color='r', linestyle='--')
ax4.set_xlabel('Residuals')
ax4.set_ylabel('Frequency')
ax4.set_title('Distribution of Residuals')

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('tabpfn_regression_results.png', dpi=150)
plt.show()

# Model Performance Comparison
print("\nModel Performance Comparison:")
model_names = []
mse_scores = []
r2_scores = []
mae_scores = []

for name, model in models.items():
    y_model_pred = model.predict(X_test)
    
    model_mse = mean_squared_error(y_test, y_model_pred)
    model_r2 = r2_score(y_test, y_model_pred)
    model_mae = mean_absolute_error(y_test, y_model_pred)
    
    model_names.append(name)
    mse_scores.append(model_mse)
    r2_scores.append(model_r2)
    mae_scores.append(model_mae)
    
    print(f"{name} - MSE: {model_mse:.4f}, R²: {model_r2:.4f}, MAE: {model_mae:.4f}")

# Plot model comparison
plt.figure(figsize=(15, 10))

# 1. MSE Comparison
plt.subplot(2, 2, 1)
plt.bar(model_names, mse_scores, color='skyblue')
plt.title('Mean Squared Error (MSE)')
plt.xticks(rotation=45)
for i, v in enumerate(mse_scores):
    plt.text(i, v + 0.1, f'{v:.3f}', ha='center')

# 2. R² Comparison
plt.subplot(2, 2, 2)
plt.bar(model_names, r2_scores, color='lightgreen')
plt.title('R² Score')
plt.xticks(rotation=45)
for i, v in enumerate(r2_scores):
    plt.text(i, v + 0.01, f'{v:.3f}', ha='center')

# 3. MAE Comparison
plt.subplot(2, 2, 3)
plt.bar(model_names, mae_scores, color='salmon')
plt.title('Mean Absolute Error (MAE)')
plt.xticks(rotation=45)
for i, v in enumerate(mae_scores):
    plt.text(i, v + 0.1, f'{v:.3f}', ha='center')

# 4. Feature Correlation Heatmap
plt.subplot(2, 2, 4)
# Combine features and target into one dataframe
df_corr = pd.concat([X_train.reset_index(drop=True), 
                     pd.Series(y_train.values, name='target').reset_index(drop=True)], axis=1)
# Calculate and plot correlation matrix
corr_matrix = df_corr.corr()
sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Feature Correlation Heatmap')

plt.tight_layout()
plt.savefig('model_comparison_regression.png', dpi=150)
plt.show()

# Feature analysis for Bayesian Optimization preparation
# This helps identify which parameters might be most important to optimize
plt.figure(figsize=(12, 8))
sorted_features_idx = np.argsort(feature_importance)
plt.barh(range(len(feature_names)), feature_importance[sorted_features_idx])
plt.yticks(range(len(feature_names)), [feature_names[i] for i in sorted_features_idx])
plt.xlabel('Feature Importance Score')
plt.title('All Features Ranked by Importance (for Bayesian Optimization)')
plt.tight_layout()
plt.savefig('feature_importance_for_bo.png', dpi=150)
plt.show()

# Prepare data for potential hyperparameter space exploration in Bayesian Optimization
print("\nPotential hyperparameter ranges for Bayesian Optimization:")
for i, feature in enumerate(feature_names):
    feature_min = X[feature].min()
    feature_max = X[feature].max()
    feature_importance_score = feature_importance[i]
    
    if feature_importance_score > np.mean(feature_importance):
        print(f"{feature}: Range [{feature_min:.4f}, {feature_max:.4f}], Importance: {feature_importance_score:.4f}")