import pickle
from re import L
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import pickle




def load_model_sarimax():
    path = 'data/sarimax-monthly.pickle'
    model = pickle.load(open(path, 'rb'))
    return model
