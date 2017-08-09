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

# Initialize the housing prices and housing features
housing_prices = city_data.target
housing_features = city_data.data
print("Boston Housing dataset loaded successfully!")

# Statistical Analysis and Data Exploration


# Number of houses in the dataset
total_houses = housing_features.shape[0]
# Number of features in the dataset
total_features = housing_features.shape[1]
# Minimum housing value in the dataset
minimum_price = np.amin(housing_prices)
# Maximum housing value in the dataset
maximum_price = np.amax(housing_prices)
# Mean house value of the dataset
mean_price = np.mean(housing_prices)
# Median house value of the dataset
median_price = np.median(housing_prices)
# Standard deviation of housing values of the dataset
std_dev = np.std(housing_prices)

# Show the calculated statistics
print "Boston Housing dataset statistics (in $1000's):"
print "Total number of houses:", total_houses
print "Total number of features:", total_features
print "Minimum house price:", minimum_price
print "Maximum house price:", maximum_price
print "Mean house price: {0:.3f}".format(mean_price)
print "Median house price:", median_price
print "Standard deviation of house price: {0:.3f}".format(std_dev)
