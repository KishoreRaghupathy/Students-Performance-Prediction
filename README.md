# Student Performance Prediction

This project aims to analyze student performance data and build machine learning models to predict student scores in math, reading, and writing. The models are built using the Gradient Boosting algorithm and compared with other regression models.

## Table of Contents
- [Installation](#installation)
- [Data](#data)
- [Model Building](#model-building)
- [Model Evaluation](#model-evaluation)
- [Comparing Models](#comparing-models)
- [Conclusion](#conclusion)
- [Contributing](#contributing)
- [License](#license)
## Workflow
<img src="/work%20flow.jpg" alt="workflow">
## Installation

To install the required packages, run:

```bash
pip install pandas numpy plotly scikit-learn
```

## Data

The dataset used in this project is `performance.csv`, which contains the following columns:
- `gender`
- `race/ethnicity`
- `parental level of education`
- `lunch`
- `test preparation course`
- `math score`
- `reading score`
- `writing score`



## Model Building

The primary model used is the Gradient Boosting Regressor. Grid Search is used to find the best hyperparameters.

```python
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor

# Define the parameter grid
param_grid = {
    'learning_rate': [0.01, 0.1, 0.2],
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 4, 5],
    'subsample': [0.8, 1.0],
    'min_samples_split': [2, 5, 10]
}

# Initialize the Gradient Boosting Regressor
gbr = GradientBoostingRegressor(random_state=42)

# Initialize the Grid Search
grid_search = GridSearchCV(estimator=gbr, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)

# Fit the Grid Search
grid_search.fit(X_train, y_train)

# Get the best parameters
best_params = grid_search.best_params_
print(f"Best parameters: {best_params}")
```

## Model Evaluation

The best model is evaluated on the test set.

```python
from sklearn.metrics import mean_squared_error, r2_score

# Make predictions with the best model
y_pred = grid_search.best_estimator_.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Best model MSE: {mse}, R²: {r2}")
```

## Comparing Models

Different models are compared based on their performance.

```python
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor

# Initialize models
models = {
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
    'Ridge Regression': Ridge(),
    'SVR': SVR()
}

# Train and evaluate models
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"{name} - MSE: {mse}, R2: {r2}")
```

## Conclusion

This project demonstrates the process of building, tuning, and evaluating machine learning models to predict student performance. The Gradient Boosting Regressor with hyperparameter tuning using Grid Search provided the best results.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any suggestions or improvements.

## License

This project is licensed under the MIT License.
```

## Screenshots

Add relevant images to the `images` directory and include them in the README using markdown image syntax. Adjust paths as necessary based on your project structure.


"# Students-Performance-Prediction" 
