from pyexpat import model
from tkinter import E
import xgboost as xgb
import numpy as  np
from sklearn.metrics import mean_absolute_percentage_error
import pandas as pd

def initialize_xgb_model(n_estimators=1000,
                         max_depth = None,
                         verbosity = 2,learning_rate = learning_rate) -> Model:
g

    model = xgb.XGBRegressor(n_estimators=n_estimators,
                           max_depth = max_depth,
                           verbosity = verbosity,
                           learning_rate = learning_rate)

    return model


def train_model(model : Model,
                X_train: np.ndarray,
                y_train: np.ndarray,
                X_test: np.ndarray,
                y_test: np.ndarray,
                early_stopping_rounds: int
                verbose = 2):

    model.fit(X_train, y_train,
              eval_set=[(X_train, y_train),(X_test, y_test)],
              early_stopping_rounds=early_stopping_rounds,
              verbose=verbose)


    return model

def score(model: model, X_test : np.ndarray, y_test):

    prediction = pd.Series(model.predict(X_test))

    return mean_absolute_percentage_error(y_test,prediction)
