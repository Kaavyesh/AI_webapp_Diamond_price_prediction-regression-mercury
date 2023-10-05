# Diamond Price Prediction

Diamond price prediction refers to the use of machine learning techniques to predict the price of a diamond based on its characteristics. By using advanced algorithms and predictive models, jewelers can streamline their pricing processes and make informed decisions for the benefit of both sellers and buyers.

## Dataset

The dataset used in this analysis contains the following columns:

- `carat`: weight of the diamond (0.2--5.01)
- `cut`: quality of the cut (Fair, Good, Very Good, Premium, Ideal)
- `color`: diamond colour, from J (worst) to D (best)
- `clarity`: a measurement of how clear the diamond is (I1 (worst), SI1, SI2, VS1, VS2, VVS1, VVS2, IF (best))
- `depth`: total depth percentage = z / mean(x, y) = 2 * z / (x + y) (43--79)
- `table`: width of top of diamond relative to widest point (43--95)
- `price`: price in US dollars (\$326--\$18,823)
- `x`: length in mm (0--10.74)
- `y`: width in mm (0--58.9)
- `z`: depth in mm (0--31.8)

## Analysis Steps

The project involves the following key steps:

1. Data Preprocessing: We load the dataset, handle missing data, and explore the data to understand its structure.

2. Feature Engineering: We may perform feature engineering if necessary, such as encoding categorical variables.

3. Model Building: We build machine learning models to predict diamond prices based on the dataset.

4. Model Evaluation: We evaluate the models using appropriate metrics such as mean absolute error, mean squared error, and R-squared.

5. Model Selection: We select the best-performing model for diamond price prediction.

6. Interpretation: We provide insights into the factors that influence diamond prices.

## Code

To build and evaluate the models, we can use the following code:

```python
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Define the models
models = [('Linear Regression', LinearRegression()),
          ('Lasso Regression', Lasso()),
          ('AdaBoost Regression', AdaBoostRegressor()),
          ('Ridge Regression', Ridge()),
          ('Gradient Boosting Regression', GradientBoostingRegressor()),
          ('Random Forest Regression', RandomForestRegressor()),
          ('KNeighbours Regression', KNeighborsRegressor())]

# Create the pipeline for each model
pipelines = []
for name, model in models:
    pipelines.append((name, Pipeline([('model', model)])))

# Fit and evaluate each model
for name, pipeline in pipelines:
    pipeline.fit(x_train, y_train)
    y_pred = pipeline.predict(x_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"{name}:")
    print(f"MAE: {mae:.2f}")
    print(f"MSE: {mse:.2f}")
    print(f"R-squared: {r2:.2f}")
    print("\n")
```

## Usage

To run the analysis or reproduce the results, follow the steps outlined in the Jupyter notebooks provided in the `notebooks/` directory. You may need to install required Python libraries using `pip` as mentioned in the notebooks.

## Contributing

If you have suggestions, improvements, or want to contribute to this project, please feel free to open an issue or submit a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

