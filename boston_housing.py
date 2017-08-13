## Model Evaluation & Validation
## Project 1: Predicting Boston Housing Prices

# Importing a few necessary libraries
import numpy as np
import matplotlib.pyplot as pl
from sklearn import datasets
from sklearn.tree import DecisionTreeRegressor
import pandas as pd
import sklearn
from sklearn.metrics import r2_score, make_scorer
from sklearn.model_selection import GridSearchCV

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

## Implement code so that the `shuffle_split_data` function does the following
## Randomly shuffle the input data `X` and target labels (housing values) `y`
## Split the data into training and testing subsets, holding 30% of the data for testin

def shuffle_split_data(X, y):
    df = pd.DataFrame(X)
    # create training and testing vars
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(df, y, test_size=0.3)
    return X_train, y_train, X_test, y_test

try:
    X_train, y_train, X_test, y_test = shuffle_split_data(housing_features, housing_prices)
    print "Successfully shuffled and split the data!"
except:
    print "Something went wrong with shuffling and splitting the data."

#Perform a total error calculation between the true values of the `y` labels `y_true` and the predicted values of the `y` labels `y_predict`.
def performance_metric(y_true, y_predict):
    error = r2_score(y_true, y_predict)
    return error
# Test performance_metric
try:
    total_error = performance_metric(y_train, y_train)
    print "Value of r2=",total_error
    print "Successfully performed a metric calculation!"
except:
    print "Something went wrong with performing a metric calculation."

#Tunes a decision tree regressor model using GridSearchCV on the input data X,
#and target labels y and returns this optimal model.

regressor = DecisionTreeRegressor()
parameters = {'max_depth':(1,2,3,4,5,6,7,8,9,10)}
scoring_function = make_scorer(r2_score)
reg = GridSearchCV(regressor, parameters, scoring = scoring_function)
reg.fit(housing_features, housing_prices)
#m = reg.best_estimator_
def fit_model(X, y):
    # Create a decision tree regressor object
    regressor = DecisionTreeRegressor()

    # Set up the parameters we wish to tune
    parameters = {'max_depth':(1,2,3,4,5,6,7,8,9,10)}

    # Make an appropriate scoring function
    scoring_function = make_scorer(r2_score)

    # Make the GridSearchCV object
    reg = GridSearchCV(regressor, parameters, scoring = scoring_function)

    # Fit the learner to the data to obtain the optimal model with tuned parameters",
    reg.fit(X, y)

    # Return the optimal model
    return reg.best_estimator_

try:
    reg = fit_model(housing_features, housing_prices)
    print "Successfully fit a model!"
except:
    print "Something went wrong with fitting a model."

def learning_curves(X_train, y_train, X_test, y_test):
    #Calculates the performance of several models with varying sizes of training data.
    #The learning and testing error rates for each model are then plotted.
    print "Creating learning curve graphs for max_depths of 1, 3, 6, and 10"
    # Create the figure window
    fig = pl.figure(figsize=(10,8))

    # We will vary the training set size so that we have 50 different sizes
    sizes = np.rint(np.linspace(1, len(X_train), 50)).astype(int)
    train_err = np.zeros(len(sizes))
    test_err = np.zeros(len(sizes))
    
    # Create four different models based on max_depth
    for k, depth in enumerate([1,3,4,5,6,10]):
        for i, s in enumerate(sizes):
            # Setup a decision tree regressor so that it learns a tree with max_depth = depth
            regressor = DecisionTreeRegressor(max_depth = depth)
            # Fit the learner to the training data
            regressor.fit(X_train[:s], y_train[:s])
            # Find the performance on the training set
            train_err[i] = performance_metric(y_train[:s], regressor.predict(X_train[:s]))
            # Find the performance on the testing set
            test_err[i] = performance_metric(y_test, regressor.predict(X_test))
        # Subplot the learning curve graph
        ax = fig.add_subplot(3, 3, k+1)
        ax.plot(sizes, test_err, lw = 2, label = 'Testing Error')
        ax.plot(sizes, train_err, lw = 2, label = 'Training Error')
        ax.legend()
        ax.set_title('max_depth = %s'%(depth))
        ax.set_xlabel('Number of Data Points in Training Set')
        ax.set_ylabel('Total Error')
        ax.set_xlim([0, len(X_train)])
    # Visual aesthetics
    fig.suptitle('Decision Tree Regressor Learning Performances', fontsize=18, y=1.03)
    fig.tight_layout()
    fig.show()

try:
    learning_curves(X_train, y_train, X_test, y_test)
    print "Successfully plotted learning curve!"
except:
    print "Something went wrong with plotting learning curve."

