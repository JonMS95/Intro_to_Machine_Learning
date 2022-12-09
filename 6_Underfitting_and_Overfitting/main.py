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