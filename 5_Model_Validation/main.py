import pandas as pd
from constants import *
from sklearn.tree import DecisionTreeRegressor

# Apart from the ones above, some functions have to be imported additionally in order to calculate the mean absolute error (MAE) later.
from sklearn.metrics import mean_absolute_error

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