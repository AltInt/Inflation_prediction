import statsmodels.api as sm
import pickle




def load_model_sarimax():
    path = 'data/sarimax-monthly.pickle'
    model_sar = pickle.load(open(path, 'rb'))
    return model_sar
