import pickle
import statsmodels.api as sm



def load_model_arima1():
    path = 'data/arima_results1.pickle'
    model = pickle.load(open(path, 'rb'))
    return model

def load_model_arima2():
    path = 'data/arima_results2.pickle'
    model = pickle.load(open(path,'rb'))
    return model
