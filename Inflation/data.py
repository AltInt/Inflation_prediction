import pandas as pd
import numpy as np

path = 'data/final_df.csv'

def load_data(path):
    df = pd.read_csv(path, index_col = 0)
    return df
