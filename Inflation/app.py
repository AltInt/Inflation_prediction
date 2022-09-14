
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from model_v2 import load_model
from model_v2 import prediction



st.markdown('''
    # Inflation Prediction
    ## A time series analysis project
''')

model_selection = st.radio('Select a model', ('LSTM', 'SARIMAX'))


if model_selection == 'SARIMAX':
    st.write('SARIMAX model')

elif model_selection == 'LSTM':
    st.write('LSTM model')

# else:
#     st.write('XGBOOST')


if st.button('Predict!'):
    # print is visible in the server output, not in the page
    print('Prediction made!')
    st.write('Prediction made!📈')


def get_line_chart_data():

    return pd.read_csv('../data/final_df.csv')



df = get_line_chart_data().set_index('Date')

y = df['RPI']
# y1 = df['RPI_YOY']
y_test5 = y[163:164]
model6 = load_model('../data/final_prediction_graph')
test_results5 = prediction(model6)
test_results5.index = pd.to_datetime(test_results5.index)

#MAKE ACTUAL_DATA = TODAY'S RPI PRINT

#st.line_chart(df[['RPI','CPI']])

y.index = pd.to_datetime(y.index)

def plot_pred(y, y_test5, test_results5):
    fig, ax = plt.subplots()
    ax.plot(y, label = 'RPI')
    ax.scatter([test_results5.index], test_results5['test_predictions'], label='prediction')
    ax.set_title('Inflation Prediciton')
    ax.set_xlabel('Years')
    ax.set_ylabel('RPI')
    ax.legend()
    fig.show()

    return fig


fig = plot_pred(y, y_test5, test_results5)
st.pyplot(fig)
