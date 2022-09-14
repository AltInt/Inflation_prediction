
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from model_lstm import load_model_lstm, prediction_lstm




st.markdown('''
    # Inflation Prediction
    ## A time series analysis project
''')

list_models = ['LSTM', 'SARIMAX']
model_selection = st.selectbox('Select a model', list_models)

@st.cache
def predict(model_selection):

    # print is visible in the server output, not in the page
    print('Prediction made!')

    if model_selection == 'LSTM':
        model = load_model_lstm()
        prediction = prediction_lstm(model)

    elif model_selection == 'SARIMAX':
        pass

    return prediction




if st.button('Predict!'):
    prediction = predict(model_selection)

    st.markdown('## Prediction made!📈')
    st.write(prediction)

##TO DO: FORMAT OUTPUT, round pred
##Add truth value
##Add chart comparing both
##Add SARIMAX MODEL file


















# def get_line_chart_data():

#     return pd.read_csv('data/final_df.csv')



# df = get_line_chart_data().set_index('Date')

# y = df['RPI']
# # y1 = df['RPI_YOY']
# y_test5 = y[163:164]
# model6 = load_model('data/final_prediction_graph')
# test_results5 = prediction(model6)
# test_results5.index = pd.to_datetime(test_results5.index)

# #MAKE ACTUAL_DATA = TODAY'S RPI PRINT

# #st.line_chart(df[['RPI','CPI']])

# y.index = pd.to_datetime(y.index)

# def plot_pred(y, y_test5, test_results5):
#     fig, ax = plt.subplots()
#     ax.plot(y, label = 'RPI')
#     ax.scatter([test_results5.index], test_results5['test_predictions'], label='prediction')
#     ax.set_title('Inflation Prediciton')
#     ax.set_xlabel('Years')
#     ax.set_ylabel('RPI')
#     ax.legend()
#     fig.show()

#     return fig

# ADD TEXT YOUR TURN

# fig = plot_pred(y, y_test5, test_results5)
# st.pyplot(fig)
