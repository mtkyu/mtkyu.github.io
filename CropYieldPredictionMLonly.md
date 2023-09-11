# Crop Yield Prediction
This report delves into predicting crop yield based on several factors like year, rainfall, pesticide usage, temperature, country/area, and the type of crop. We assess various models to select the best performer and make predictions.

# Introduction
The Crop Yield Prediction ML Model is designed to forecast the yield of specific crops based on multiple environmental and agricultural factors. Accurate predictions of crop yields are crucial for farmers, policymakers, and supply chain stakeholders to make informed decisions about crop management, sales, and distribution.

# Objective
The primary goal of the project is to develop an ML model that can predict the yield of a crop given factors like year, average annual rainfall, quantity of pesticides used, average temperature, geographical region, and crop type.

# Data
The dataset utilized in this project encompasses several variables:

- Year: The specific year of the data entry.
- Average Rainfall: Average annual rainfall measured in mm.
- Pesticides: Total amount of pesticides used in tonnes.
- Average Temperature: Average annual temperature.
- Area: Specific region or country where the data was recorded.
- Item: The type of crop.
- Yield: The yield of the crop measured in hg/ha.

# Preprocessing
- Cleaning: The data underwent a cleaning process where duplicates and NA values were removed.
- Feature Rearrangement: Relevant features were selected and rearranged for better readability and processing.
- Categorical Data Handling: Categorical features like 'Area' and 'Item' were transformed into numerical form using one-hot encoding.
- Scaling: Continuous features like 'Year', 'Rainfall', 'Pesticides', and 'Temperature' were standardized.

# Model Selection & Training
Several regression models were trained to predict crop yield:

- Linear Regression: Basic linear model.
- Lasso Regression: Linear model with L1 regularization.
- Ridge Regression: Linear model with L2 regularization.
- K-Neighbors Regressor: Model based on 'k' nearest points in the feature space.
- Decision Tree Regressor: Tree-based model to predict yield.
After training, each model's performance was evaluated using the Mean Absolute Error (MAE) and the R^2 score. Among the models, the Decision Tree Regressor demonstrated superior performance, making it the chosen model for subsequent predictions.

# Deployment & User Interaction
A prediction function was developed, allowing users to input various factors to receive a predicted crop yield. Additionally, the trained Decision Tree Regressor and preprocessing steps were serialized (using pickle) to be deployed in a potential web application or API.

That wraps up the overview of this project. 

Now to get started: 

## 1. Data Cleaning and Exploration



```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
```


```python
df = pd.read_csv('yield_df.csv')

# Drop duplicates and NA values
df.drop_duplicates(inplace=True)
df.dropna(inplace=True)

# Ensure `Year` is in integer format
df['Year'] = df['Year'].astype(int)
```

## 2. Train, Test, Split and Rearranging Columns



```python
from sklearn.model_selection import train_test_split

col = ['Year', 'average_rain_fall_mm_per_year', 'pesticides_tonnes', 'avg_temp', 'Area', 'Item', 'hg/ha_yield']
df = df[col]
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

```

## 3. Data Preprocessing: Convert Categorical to Numerical and Scale the values



```python
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

ohe = OneHotEncoder(drop = 'first')
scaler = StandardScaler()

preprocessor = ColumnTransformer(
    transformers = [
        ('onehotencoder', ohe, [4,5]),
        ('standardization', scaler, [0,1,2,3])
    ],
    remainder = 'passthrough'
)

X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

```

## 4. Model Training, Selecting, and Prediction



```python
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, r2_score

models = {
    'lr': LinearRegression(),
    'lss': Lasso(max_iter=5000),
    'Rid': Ridge(),
    'Knr': KNeighborsRegressor(),
    'Dtr': DecisionTreeRegressor()
}

for name, model in models.items():
    model.fit(X_train_processed, y_train)
    y_pred = model.predict(X_test_processed)
    print(f"{name} : MSE : {mean_absolute_error(y_test, y_pred)} Score : {r2_score(y_test, y_pred)}")

```

    lr : MSE : 29582.438736964305 Score : 0.7551426696136498
    lss : MSE : 29565.021687814206 Score : 0.7551159177529814
    Rid : MSE : 29530.272019617714 Score : 0.7552114888565634
    Knr : MSE : 4398.097822623474 Score : 0.9849862743638015
    Dtr : MSE : 3611.831297574792 Score : 0.9795501906717161
    

## 5. Prediction Function



```python
def prediction(Year, average_rain_fall_mm_per_year, pesticides_tonnes, avg_temp, Area, Item):
    features = np.array([[Year, average_rain_fall_mm_per_year, pesticides_tonnes, avg_temp, Area, Item]], dtype=object)
    transformed_features = preprocessor.transform(features)
    predicted_yield = models['Dtr'].predict(transformed_features).reshape(1, -1)
    return predicted_yield[0]

# Sample Prediction
Year = 2000
average_rain_fall_mm_per_year = 59.0
pesticides_tonnes = 3024.11
avg_temp = 26.55
Area = 'Saudi Arabia'
Item = 'Sorghum'
result = prediction(Year, average_rain_fall_mm_per_year, pesticides_tonnes, avg_temp, Area, Item)
```

    C:\Users\mtmic\anaconda3\lib\site-packages\sklearn\base.py:420: UserWarning: X does not have valid feature names, but OneHotEncoder was fitted with feature names
      warnings.warn(
    C:\Users\mtmic\anaconda3\lib\site-packages\sklearn\base.py:420: UserWarning: X does not have valid feature names, but StandardScaler was fitted with feature names
      warnings.warn(
    


```python
result
```




    array([13384.])



## 6. Saving Models



```python
import pickle

pickle.dump(models['Dtr'], open('dtr.pkl', 'wb'))
pickle.dump(preprocessor, open('preprocessor.pkl', 'wb'))

```

# Conclusion
This workflow helped us in preprocessing data, training multiple regression models, and selecting the best model for predicting crop yields. Further refinement and validation are recommended for real-world applications.


```python

```
