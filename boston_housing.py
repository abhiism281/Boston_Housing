## Model Evaluation & Validation
## Project 1: Predicting Boston Housing Prices

# Importing a few necessary libraries
import numpy as np
import matplotlib.pyplot as pl
from sklearn import datasets
from sklearn.tree import DecisionTreeRegressor
# Make matplotlib show our plots inline (nicely formatted in the notebook)
# matplotlib inline

# Create our client's feature set for which we will be predicting a selling price
# Load the Boston Housing dataset into the city_data variable
city_data = datasets.load_boston()

# Initialize the housing prices and housing features\n
housing_prices = city_data.target
housing_features = city_data.data
print("Boston Housing dataset loaded successfully!")
