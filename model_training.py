import csv
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from datetime import datetime
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, ExpSineSquared, DotProduct, RationalQuadratic, WhiteKernel
from sklearn.svm import SVR
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor

# Create df for model
df = pd.read_csv('final_dataset.csv', delimiter=',')
df = df.drop('period_from', axis=1)
df = df.drop('period_until', axis=1)
#Date to numeric
df['Date'] = df['Date'].map(lambda x: datetime.strptime(x, "%m/%d/%Y")) 
df['Date'] = pd.to_numeric(df['Date'], errors='coerce')

# Check for NaN values
df = df.dropna()

# Set variales for Model training
X = df.drop('imbalance_price', axis=1).values
scaler = StandardScaler()
X = scaler.fit_transform(X)
y = df["imbalance_price"].values


# Define kernel for Gaussian Process
kernel = (C(1.0, (0.01, 100)) * RBF(length_scale=1.0, length_scale_bounds=(0.01, 100)) + C(1.0, (0.01, 100)) * ExpSineSquared(length_scale=1.0, periodicity=1.0, 
                                                length_scale_bounds=(0.01, 100), periodicity_bounds=(0.01, 100)) + DotProduct() ) 


gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=15, alpha=10)

# TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=5)

# Perform TSCV
mse_scores = []
mae_scores = []
rmse_scores = []
forecast_biases = []

for train_index, test_index in tscv.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    # Fit the model on training data
    gpr.fit(X_train, y_train)
    
    # Predict on the test data
    y_pred, sigma = gpr.predict(X_test, return_std=True)
    
    # Calculate metrics for this fold
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    forecast_bias = np.mean(y_pred - y_test)
    
    mse_scores.append(mse)
    mae_scores.append(mae)
    rmse_scores.append(rmse)
    forecast_biases.append(forecast_bias)

# Average metrics across all folds
average_mse = np.mean(mse_scores)
average_mae = np.mean(mae_scores)
average_rmse = np.mean(rmse_scores)
average_forecast_bias = np.mean(forecast_biases)

print(f'Average MSE: {average_mse}')
print(f'Average MAE: {average_mae}')
print(f'Average RMSE: {average_rmse}')
print(f'Average Forecast Bias: {average_forecast_bias}')
print(f'Optimized kernel: {gpr.kernel_}')


# Support Vector Machines
print("SVM")
svr = SVR(kernel='poly', C=10, epsilon=0.1)

mse_scores = []
mae_scores = []
rmse_scores = []
forecast_biases = []

for train_index, test_index in tscv.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    svr.fit(X_train, y_train)
    
    y_pred = svr.predict(X_test)
    
    # Calculate metrics 
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    forecast_bias = np.mean(y_pred - y_test)
    
    mse_scores.append(mse)
    mae_scores.append(mae)
    rmse_scores.append(rmse)
    forecast_biases.append(forecast_bias)

# Average metrics across all folds
average_mse = np.mean(mse_scores)
average_mae = np.mean(mae_scores)
average_rmse = np.mean(rmse_scores)
average_forecast_bias = np.mean(forecast_biases)

print(f'Average MSE: {average_mse}')
print(f'Average MAE: {average_mae}')
print(f'Average RMSE: {average_rmse}')
print(f'Average Forecast Bias: {average_forecast_bias}')


# Regression
print("Regression")
# Regularization
ridge = Ridge()
lasso = Lasso()

# hyperparameter grid for tuning
ridge_params = {'alpha': [0.01, 0.1, 1, 10, 100]}
lasso_params = {'alpha': [0.01, 0.1, 1, 10, 100]}

ridge_grid = GridSearchCV(ridge, ridge_params, cv=tscv, scoring='neg_mean_squared_error')
lasso_grid = GridSearchCV(lasso, lasso_params, cv=tscv, scoring='neg_mean_squared_error')

ridge_grid.fit(X, y)
ridge_best = ridge_grid.best_estimator_

lasso_grid.fit(X, y)
lasso_best = lasso_grid.best_estimator_

# Output best parameters
print("Best Ridge alpha:", ridge_grid.best_params_)
print("Best Lasso alpha:", lasso_grid.best_params_)

# Evaluate with cross-validation
models = [('Ridge', ridge_best), ('Lasso', lasso_best)]
for name, model in models:
    mse_scores = []
    mae_scores = []
    rmse_scores = []
    forecast_biases = []
    
    for train_index, test_index in tscv.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
       
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        forecast_bias = np.mean(y_pred - y_test)
        
        mse_scores.append(mse)
        mae_scores.append(mae)
        rmse_scores.append(rmse)
        forecast_biases.append(forecast_bias)
    
    # Average metrics across all folds
    average_mse = np.mean(mse_scores)
    average_mae = np.mean(mae_scores)
    average_rmse = np.mean(rmse_scores)
    average_forecast_bias = np.mean(forecast_biases)
    
    print(f'\n{name} Model Performance:')
    print(f'Average MSE: {average_mse}')
    print(f'Average MAE: {average_mae}')
    print(f'Average RMSE: {average_rmse}')
    print(f'Average Forecast Bias: {average_forecast_bias}')

    #Random forest
    print("Random Forest")
    # Define the model
rf = RandomForestRegressor()

# hyperparameter grid for tuning
param_grid = {
    'n_estimators': [10, 50, 100],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt', 'log2']
}

# GridSearchCV for RF
grid = GridSearchCV(rf, param_grid, cv=tscv, scoring='neg_mean_squared_error')

grid.fit(X, y)

# Output best parameters
print("Best Parameters:", grid.best_params_)

# Evaluate with cross-validation
mse_scores = []
mae_scores = []
rmse_scores = []
forecast_biases = []

for train_index, test_index in tscv.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    # Use the best model from grid search
    best_model = grid.best_estimator_
    
    # Fit the model on training data
    best_model.fit(X_train, y_train)
    
    # Predict on the test data
    y_pred = best_model.predict(X_test)
    
    # Calculate metrics for this fold
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    forecast_bias = np.mean(y_pred - y_test)
    
    mse_scores.append(mse)
    mae_scores.append(mae)
    rmse_scores.append(rmse)
    forecast_biases.append(forecast_bias)

# Average metrics across all folds
average_mse = np.mean(mse_scores)
average_mae = np.mean(mae_scores)
average_rmse = np.mean(rmse_scores)
average_forecast_bias = np.mean(forecast_biases)

print(f'Average MSE: {average_mse}')
print(f'Average MAE: {average_mae}')
print(f'Average RMSE: {average_rmse}')
print(f'Average Forecast Bias: {average_forecast_bias}')

# Feature Importance
feature_importance = best_model.feature_importances_
print("Feature Importance:", feature_importance)