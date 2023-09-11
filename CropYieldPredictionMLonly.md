{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "a0d82f43-893e-4b87-b82e-b936a0fa8786",
   "metadata": {},
   "source": [
    "# Crop Yield Prediction\n",
    "This report delves into predicting crop yield based on several factors like year, rainfall, pesticide usage, temperature, country/area, and the type of crop. We assess various models to select the best performer and make predictions."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "4fe1d799-dcdc-45a0-806f-053eed2b8e12",
   "metadata": {},
   "source": [
    "# Introduction\n",
    "The Crop Yield Prediction ML Model is designed to forecast the yield of specific crops based on multiple environmental and agricultural factors. Accurate predictions of crop yields are crucial for farmers, policymakers, and supply chain stakeholders to make informed decisions about crop management, sales, and distribution.\n",
    "\n",
    "# Objective\n",
    "The primary goal of the project is to develop an ML model that can predict the yield of a crop given factors like year, average annual rainfall, quantity of pesticides used, average temperature, geographical region, and crop type.\n",
    "\n",
    "# Data\n",
    "The dataset utilized in this project encompasses several variables:\n",
    "\n",
    "- Year: The specific year of the data entry.\n",
    "- Average Rainfall: Average annual rainfall measured in mm.\n",
    "- Pesticides: Total amount of pesticides used in tonnes.\n",
    "- Average Temperature: Average annual temperature.\n",
    "- Area: Specific region or country where the data was recorded.\n",
    "- Item: The type of crop.\n",
    "- Yield: The yield of the crop measured in hg/ha.\n",
    "\n",
    "# Preprocessing\n",
    "- Cleaning: The data underwent a cleaning process where duplicates and NA values were removed.\n",
    "- Feature Rearrangement: Relevant features were selected and rearranged for better readability and processing.\n",
    "- Categorical Data Handling: Categorical features like 'Area' and 'Item' were transformed into numerical form using one-hot encoding.\n",
    "- Scaling: Continuous features like 'Year', 'Rainfall', 'Pesticides', and 'Temperature' were standardized.\n",
    "\n",
    "# Model Selection & Training\n",
    "Several regression models were trained to predict crop yield:\n",
    "\n",
    "- Linear Regression: Basic linear model.\n",
    "- Lasso Regression: Linear model with L1 regularization.\n",
    "- Ridge Regression: Linear model with L2 regularization.\n",
    "- K-Neighbors Regressor: Model based on 'k' nearest points in the feature space.\n",
    "- Decision Tree Regressor: Tree-based model to predict yield.\n",
    "After training, each model's performance was evaluated using the Mean Absolute Error (MAE) and the R^2 score. Among the models, the Decision Tree Regressor demonstrated superior performance, making it the chosen model for subsequent predictions.\n",
    "\n",
    "# Deployment & User Interaction\n",
    "A prediction function was developed, allowing users to input various factors to receive a predicted crop yield. Additionally, the trained Decision Tree Regressor and preprocessing steps were serialized (using pickle) to be deployed in a potential web application or API.\n",
    "\n",
    "That wraps up the overview of this project. \n",
    "\n",
    "Now to get started: "
   ]
  },
  {
   "cell_type": "markdown",
   "id": "a5c54180-354c-4463-9d52-8c1ba7b913b3",
   "metadata": {},
   "source": [
    "## 1. Data Cleaning and Exploration\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "c8c6e348-1d4f-41fb-ada1-265434e68e3c",
   "metadata": {
    "tags": []
   },
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "f89f3a97-de57-4b8d-80c2-151428f9c024",
   "metadata": {
    "tags": []
   },
   "outputs": [],
   "source": [
    "df = pd.read_csv('yield_df.csv')\n",
    "\n",
    "# Drop duplicates and NA values\n",
    "df.drop_duplicates(inplace=True)\n",
    "df.dropna(inplace=True)\n",
    "\n",
    "# Ensure `Year` is in integer format\n",
    "df['Year'] = df['Year'].astype(int)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "2f056eb8-95fe-4535-8728-792ff4cb3b86",
   "metadata": {},
   "source": [
    "## 2. Train, Test, Split and Rearranging Columns\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "b3d1f0c7-64d1-4d2a-aa50-a78cdceaa378",
   "metadata": {
    "tags": []
   },
   "outputs": [],
   "source": [
    "from sklearn.model_selection import train_test_split\n",
    "\n",
    "col = ['Year', 'average_rain_fall_mm_per_year', 'pesticides_tonnes', 'avg_temp', 'Area', 'Item', 'hg/ha_yield']\n",
    "df = df[col]\n",
    "X = df.iloc[:, :-1]\n",
    "y = df.iloc[:, -1]\n",
    "\n",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "5816572e-4c51-475d-8015-dd15561eee5f",
   "metadata": {},
   "source": [
    "## 3. Data Preprocessing: Convert Categorical to Numerical and Scale the values\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "9425902d-2ffa-48dc-bb97-4bc7ebd845b1",
   "metadata": {
    "tags": []
   },
   "outputs": [],
   "source": [
    "from sklearn.preprocessing import OneHotEncoder, StandardScaler\n",
    "from sklearn.compose import ColumnTransformer\n",
    "\n",
    "ohe = OneHotEncoder(drop = 'first')\n",
    "scaler = StandardScaler()\n",
    "\n",
    "preprocessor = ColumnTransformer(\n",
    "    transformers = [\n",
    "        ('onehotencoder', ohe, [4,5]),\n",
    "        ('standardization', scaler, [0,1,2,3])\n",
    "    ],\n",
    "    remainder = 'passthrough'\n",
    ")\n",
    "\n",
    "X_train_processed = preprocessor.fit_transform(X_train)\n",
    "X_test_processed = preprocessor.transform(X_test)\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "183cd132-180c-4df8-8b98-fd7de64cdadd",
   "metadata": {},
   "source": [
    "## 4. Model Training, Selecting, and Prediction\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "f1553a93-a9e2-47c2-b3a6-215eef485a40",
   "metadata": {
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "lr : MSE : 29582.438736964305 Score : 0.7551426696136498\n",
      "lss : MSE : 29565.021687814206 Score : 0.7551159177529814\n",
      "Rid : MSE : 29530.272019617714 Score : 0.7552114888565634\n",
      "Knr : MSE : 4398.097822623474 Score : 0.9849862743638015\n",
      "Dtr : MSE : 3611.831297574792 Score : 0.9795501906717161\n"
     ]
    }
   ],
   "source": [
    "from sklearn.linear_model import LinearRegression, Lasso, Ridge\n",
    "from sklearn.neighbors import KNeighborsRegressor\n",
    "from sklearn.tree import DecisionTreeRegressor\n",
    "from sklearn.metrics import mean_absolute_error, r2_score\n",
    "\n",
    "models = {\n",
    "    'lr': LinearRegression(),\n",
    "    'lss': Lasso(max_iter=5000),\n",
    "    'Rid': Ridge(),\n",
    "    'Knr': KNeighborsRegressor(),\n",
    "    'Dtr': DecisionTreeRegressor()\n",
    "}\n",
    "\n",
    "for name, model in models.items():\n",
    "    model.fit(X_train_processed, y_train)\n",
    "    y_pred = model.predict(X_test_processed)\n",
    "    print(f\"{name} : MSE : {mean_absolute_error(y_test, y_pred)} Score : {r2_score(y_test, y_pred)}\")\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "4ed9c1e4-6a34-4a20-a3a1-e3597c59e47a",
   "metadata": {
    "tags": []
   },
   "source": [
    "## 5. Prediction Function\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "99511f3a-68c4-4d09-89e2-6972f4842d1e",
   "metadata": {
    "tags": []
   },
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\mtmic\\anaconda3\\lib\\site-packages\\sklearn\\base.py:420: UserWarning: X does not have valid feature names, but OneHotEncoder was fitted with feature names\n",
      "  warnings.warn(\n",
      "C:\\Users\\mtmic\\anaconda3\\lib\\site-packages\\sklearn\\base.py:420: UserWarning: X does not have valid feature names, but StandardScaler was fitted with feature names\n",
      "  warnings.warn(\n"
     ]
    }
   ],
   "source": [
    "def prediction(Year, average_rain_fall_mm_per_year, pesticides_tonnes, avg_temp, Area, Item):\n",
    "    features = np.array([[Year, average_rain_fall_mm_per_year, pesticides_tonnes, avg_temp, Area, Item]], dtype=object)\n",
    "    transformed_features = preprocessor.transform(features)\n",
    "    predicted_yield = models['Dtr'].predict(transformed_features).reshape(1, -1)\n",
    "    return predicted_yield[0]\n",
    "\n",
    "# Sample Prediction\n",
    "Year = 2000\n",
    "average_rain_fall_mm_per_year = 59.0\n",
    "pesticides_tonnes = 3024.11\n",
    "avg_temp = 26.55\n",
    "Area = 'Saudi Arabia'\n",
    "Item = 'Sorghum'\n",
    "result = prediction(Year, average_rain_fall_mm_per_year, pesticides_tonnes, avg_temp, Area, Item)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "3012014d-c4b9-4efa-a48c-480b887b4456",
   "metadata": {
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([13384.])"
      ]
     },
     "execution_count": 11,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "result"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "ccca46bc-8d14-44f5-8792-11ffa7910a4c",
   "metadata": {},
   "source": [
    "## 6. Saving Models\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "8cfb1f52-1b95-4678-b8fb-89fd29f7b936",
   "metadata": {
    "tags": []
   },
   "outputs": [],
   "source": [
    "import pickle\n",
    "\n",
    "pickle.dump(models['Dtr'], open('dtr.pkl', 'wb'))\n",
    "pickle.dump(preprocessor, open('preprocessor.pkl', 'wb'))\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "63de0624-5d3b-4749-a568-9e110a1e3923",
   "metadata": {},
   "source": [
    "# Conclusion\n",
    "This workflow helped us in preprocessing data, training multiple regression models, and selecting the best model for predicting crop yields. Further refinement and validation are recommended for real-world applications."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "cb02aaf9-f9db-403c-9f9c-bb31fe7ec143",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.10.9"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
