#setup
import math
from matplotlib import cm
from matplotlib import gridspec
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.python.data import Dataset

tf.logging.set_verbosity(tf.logging.ERROR)
pd.set_option('display.max_row', 10)
pd.set_option('display.float_format', '{: .1f}'.format)

#load data
california_housing_dataframe = pd.read_csv(
    "https://storage.googleapis.com/mledu-datasets/california_housing_train.csv", sep=",")
california_housing_dataframe = california_housing_dataframe.reindex(
    np.random.permutation(california_housing_dataframe.index)
)
california_housing_dataframe['median_house_value'] /= 1000.0
california_housing_dataframe
#examine the data
california_housing_dataframe.describe()

#build the first model

#stip1:define Feature and configure Feature Columns
#define the input feature: total_rooms
my_feature = california_housing_dataframe[["total_rooms"]]

#configure a numeric feature column for total_rooms
feature_columns = [tf.feature_column.numeric_column('total_rooms')]

