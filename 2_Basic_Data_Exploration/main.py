# First of all, pandas library should be imported. This one is one of the most known libraries in data analysis.
import pandas as pd
from constants import *

# Basically, this reads a comma-separated values file as two-dimensional data structure with labeled axes.
melbourne_data = pd.read_csv(PATH_MELBOURNE_CSV)

# Now, we want to getsome statistical information. By using the 'describe' method, we can get some summarized data,
# such as count, mean, std, max value and percentiles.
melbourne_data_frame = melbourne_data.describe()

# Print full CSV file
print(TEXT_SEPARATOR)
print(TEXT_FULL_CSV)
print(melbourne_data)

# Print data frame statistical description
print(TEXT_SEPARATOR)
print(TEXT_DESCR)
print(melbourne_data_frame)

# Meaning of the rows in the description variable:
# count: the total amount of data objects.
# mean: mean value of each column.
# std: standard deviation, i.e., how far are the values from the main value.
# min: minimum value.
# 25%, 50%, 75%: percentiles, it's to say, which value is above the % shown.
# max: maximum value