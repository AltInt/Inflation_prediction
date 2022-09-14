from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import BatchNormalization
from tensorflow import expand_dims
from tensorflow.keras.losses import MeanAbsolutePercentageError
import tensorflow
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from data import load_data

path = '../data/final_prediction_graph'

def load_model(path):
    model = tensorflow.keras.models.load_model(path)
    return model

def prediction(model):
    path2 = '../data/final_df.csv'
    df = load_data(path2)
    y=df['RPI']
    X = df.drop(columns = ['RPI', 'CPI', 'RPI YOY'])
    X_test5, y_test5 = X[163:164], y[163:164]
    X_test5_x = expand_dims(X_test5, -2)
    test_prediction5 = list(model.predict(X_test5_x))
    test_results5 = pd.DataFrame(columns = ['test_predictions', 'test_actual'])
    test_results5['date'] = y_test5.index
    test_results5['test_predictions'] = [x[0] for x in test_prediction5]
    test_results5.set_index('date', inplace=True)

    return test_results5


# path2 = '../data/final_df.csv'
# df = load_data(path2)

prediction(load_model(path))
