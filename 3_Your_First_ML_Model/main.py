import pandas as pd
from constants import *
from sklearn.tree import DecisionTreeRegressor

#Read the CSV file into a variable.
melbourne_data = pd.read_csv(PATH_MELBOURNE_CSV)

# Store in list the names of the columns of the previously read CSV file, then print the list.
melbourne_data_cols = melbourne_data.columns
print(TEXT_SEPARATOR)
print(TEXT_COLS)
print(melbourne_data_cols)

# The used data may have some values marked as NA. Axis = 0 (or simply 0) is used to erase rows, while 1 is used to erase columns.
# By now, every column that contains any NA value will be removed.
melbourne_data = melbourne_data.dropna(axis = 0)

# To select the subset of data we are looking forward to work with, there are two main options:
#     ·Dot notation, which is used to select the prediction target.
#     ·Selecting a column, used to select the features.
# Columns are stored within series, which is something like a DataFrame that contains a single type of data.

# In this case, price is going to be our prediction target, so let's store that series into a variable.
y = melbourne_data.Price

# Trivia fact: the name of the target variable is usually named 'y', as it's the variable that's going to depend on others.

# Now, let's select the features, in other words, the data we are meant to use in order to predict our target variable ('y').
# As a convention, it's called 'X'.
# First, store in a list the names of the columns we are looking for, then store those in the 'X' variable.
melbourne_features = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longitude']
X = melbourne_data[melbourne_features]

# Store the data frame (stats) of the target features.
X_data_frame = X.describe()

print('\n' + TEXT_SEPARATOR)
print(TEXT_FEATURES)
print(X_data_frame)

# Use the "head" function to get some of the first rows in a data frame. By default, the input parameter is 5.
# We will use the data stored within the first five rows to make our first prediction.
X_first_five_rows = X.head()

# It's time to make the prediction model. Before building it, let's enumerate the steps that should be followed:
#    ·Define: choose the predcition model that's going to be used. By now, only decision tree has been explained.
#    ·Fit: capture patterns from provided data.
#    ·Predict.
#    ·Evaluate: define how accurate the predictions done by the model are.

# Define the model. As seen in the imports in the header of this file, DTR was imported.
# Watch this video if you want to have some more theoretical explanation about how does DTR work:
# https://www.youtube.com/watch?v=UhY5vPfQIrA
melbourne_model = DecisionTreeRegressor(random_state = DTR_RANDOM_STATE)

# Fit the model:
melbourne_model.fit(X, y)

# Check the model:
print('\n' + TEXT_SEPARATOR)
print(TEXT_FEAT_FIVE_ROWS)
print(X_first_five_rows)

melbourne_price_predictions = melbourne_model.predict(X)
print('\n' + TEXT_SEPARATOR)
print(TEXT_PREDICTIONS)
print(melbourne_price_predictions)

print('\n' + TEXT_SEPARATOR)
print(TEXT_FIVE_TARGET_PRICES)
print(y.head())
print('\n' + TEXT_SEPARATOR)
print(TEXT_FIVE_PRED_PRICES)
print(melbourne_price_predictions[0:5])

print('\n' + TEXT_SEPARATOR)
print(TEXT_OUTCOME)