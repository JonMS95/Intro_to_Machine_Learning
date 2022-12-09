import pandas as pd
from constants import *
from sklearn.tree import DecisionTreeRegressor

# Apart from the ones above, some functions have to be imported additionally in order to calculate the mean absolute error (MAE) later.
from sklearn.metrics import mean_absolute_error

# The steps taht are followed are pretty much the same as in the last lesson except for the ones after the prediction. Let's enumerate them below:
#     1-Get the data from the CSV file.
#     2-Store the column names somewhere.
#     3-Remove every row that contains a null value (N/A).
#     4-Store the price column series in a variable called 'y'.
#     5-Store in 'X' the name of all the columns that are meant to be used as features, it's to say, the ones that are going to be used to build the model.
#     6-Get the series (columns of the source CSV file) that were stored in the previous step.
#     7-Get the description of those, i.e., the stats of the data in the X variable.
#     8-Store somewhere else the values of the first five data rows of X.
#     9-Instantiate an object of the DecisionTreeRegressor class by using a predefined random state.
#     10-Generate the model by using the 'fit' method of the DecisionTreeRegressor class.
#     11-Generate the prediction in order to get y from X. If the model is good, the returned values should be the same as the exepected ones when using same data.
#     12-Calculate the MAE by using the 'mean_absolute_error' method.

melbourne_data = pd.read_csv(PATH_MELBOURNE_CSV)
melbourne_data_cols = melbourne_data.columns
melbourne_data = melbourne_data.dropna(axis = 0)

y = melbourne_data.Price

melbourne_features = ['Rooms', 'Bathroom', 'Landsize', 'BuildingArea', 'YearBuilt', 'Lattitude', 'Longitude']
X = melbourne_data[melbourne_features]

X_data_frame = X.describe()
X_first_five_rows = X.head()

melbourne_model = DecisionTreeRegressor(random_state = DTR_RANDOM_STATE)
melbourne_model.fit(X, y)

melbourne_predict_prices = melbourne_model.predict(X)
melbourne_MAE = mean_absolute_error(y, melbourne_predict_prices)

print(melbourne_MAE)

# Until this point, it can be noticed that mean absolute error is unsurprisingly low. It's something expectable, as the data that was used first to train the model
# is the same as the data that's being used later to test it, which is quite unfair, to be honest.