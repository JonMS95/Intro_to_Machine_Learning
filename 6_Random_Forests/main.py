# Decision trees leave us with a difficult decision. A deep tree with lots of leaves will overfit because each prediction is coming from historical data from
# only the few houses at its leaf. But a shallow tree with few leaves will perform poorly because it fails to capture as many distinctions in the raw data.
# Random forest is an alternative technique that uses many trees, and then it makes predctions by averaging the predictions of each component tree.
# It generally has much better performance than a single decision tree and it usually works well with default parameters.

from constants import *
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor

melbourne_data = pd.read_csv(PATH_MELBOURNE_CSV)
melbourne_data = melbourne_data.dropna(axis = 0)

y = melbourne_data.Price

melbourne_features = ['Rooms', 'Bathroom', 'Landsize', 'BuildingArea', 'YearBuilt', 'Lattitude', 'Longitude']
X = melbourne_data[melbourne_features]

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = SPLIT_RANDOM_STATE)

forest_model = RandomForestRegressor(random_state = RFR_RANDOM_STATE)
forest_model.fit(train_X, train_y)

melb_preds = forest_model.predict(val_X)
RFR_MAE = mean_absolute_error(melb_preds, val_y)
print("Random forest model MAE: {:,.2f}".format(RFR_MAE))