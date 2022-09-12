import pandas as pd
import numpy as np

def load_data():
    df = pd.read_csv('../raw_data/final_df.csv')
    return df
