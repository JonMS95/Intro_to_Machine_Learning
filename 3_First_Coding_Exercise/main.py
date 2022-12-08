import pandas as pd
from constants import *

iowa_data = pd.read_csv(PATH_IOWA_CSV)

iowa_data_frame = iowa_data.describe()