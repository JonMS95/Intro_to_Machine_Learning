# As it's known for the average software engineer, a decission tree (same as a binary tree) has 'leaves', which are the nodes that don't have any other nodes
# coming out from them, it's to say, those nodes do not have any child. The more times the nodes get split into to other nodes, the bigger the amount of leaves
# in the tree it's going to be. The number of splits determines the number of leaves, which follows the next equation: num_leaves = 2^(num_of_splits). Thus, if
# 10 splits were made, then the tree is going to have 1024 leaves.

# Using more or less leaves may make the model more or less reliable in different situations. Two phenomena may occur here:
#     ·Overfitting: there are too many leaves in the tree. Therefore, the model makes realy accurate predictions for the training data, but its performance is
#     very poor when it has to predict any target value by using validation data (it's to say, data from outside of the training data set). For example, this
#     may happen if for the previous exercises, a very low amount of houses is associated to each leaf, which leads the tree to have too many leaves.
#     ·Underfitting: there are too many data points in each leave. Therefore, the amount of leaves in the tree is not enough for the model to make a trustworthy
#     prediction. This leads the model to not be able to make good predictions even with training data, which makes it even more unreliable with validation data.

# It's worth it pointing out that the accuracy of the model is not necessarily proportional to the number of leaves in the tree, because it may lead the model to
# overfit when using validation data. That's why it's strongly recommended to calculate the mean absolute error for a generous number of trees, each one having a
# different number of leaves within it.

from constants import *
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split

# The function below takes some training and validation data, the maximum amount of leaves for the decision tree that's meant to be trained and validated, and the
# random state used in that same decision tree.
def get_MAE(input_max_leaf_nodes, input_random_state, train_X, val_X, train_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes = input_max_leaf_nodes, random_state = input_random_state)
    model.fit(train_X, train_y)
    prediction = model.predict(val_X)
    MAE = mean_absolute_error(prediction, val_y)
    return MAE

melbourne_data = pd.read_csv(PATH_MELBOURNE_CSV)
melbourne_data_cols = melbourne_data.columns
melbourne_data = melbourne_data.dropna(axis = 0)

y = melbourne_data.Price

melbourne_features = ['Rooms', 'Bathroom', 'Landsize', 'BuildingArea', 'YearBuilt', 'Lattitude', 'Longitude']
X = melbourne_data[melbourne_features]

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = SPLIT_RANDOM_STATE)

calculated_MAE = {}

for leaves in MAX_LEAF_NODES:
    MAE = int(get_MAE(leaves, DTR_RANDOM_STATE, train_X, val_X, train_y, val_y))
    if MAE not in calculated_MAE.keys():
        calculated_MAE[MAE] = [leaves]
    else:
        calculated_MAE[MAE].append(leaves)
    print(TEXT_SEPARATOR)
    print("Number of leaves: " + str(leaves) + "\tMAE: " + str(MAE))

best_leaves_number = (calculated_MAE[min(calculated_MAE.keys())])[0]
print(TEXT_SEPARATOR)
print("BEST PERFORMING MODEL\nMAE: " + str(min(calculated_MAE.keys())) + "\nNumber of leaves: " + str(calculated_MAE[min(calculated_MAE.keys())]))

# As it can be noticed after executing the code, the model that behaves the best is the one that uses 500 leaves. Now that the optimal number of leaves for the
# tree is known, we can just take all the data (whole X and y variables), and use them again to train the model.

final_model = DecisionTreeRegressor(max_leaf_nodes = best_leaves_number, random_state = DTR_RANDOM_STATE)
final_model.fit(X, y)