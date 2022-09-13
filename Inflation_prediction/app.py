
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.markdown('''
    # Inflation Prediction
    ## A time series analysis project
''')

model_selection = st.radio('Select a model', ('SARIMAX','LSTM','XGBOOST'))

st.write(model_selection)

if model_selection == 'SARIMAX':
    st.write('SARIMAX model')

elif model_selection == 'LSTM':
    st.write('LSTM model')

else:
    st.write('XGBOOST')


if st.button('Predict!'):
    # print is visible in the server output, not in the page
    print('Prediction made!')
    st.write('Prediction made!📈')


def get_line_chart_data():

    return pd.read_csv('../raw_data/final_df.csv')

df = get_line_chart_data()

fig, ax = plt.subplots()

ax.plot(df['RPI'])

st.pyplot(fig)



#st.line_chart(df[['RPI','CPI']])
