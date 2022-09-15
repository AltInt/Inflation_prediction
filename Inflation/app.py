import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from Inflation.model_arima import load_model_arima1
from model_lstm import load_model_lstm, prediction_lstm
from model_sarimax import load_model_sarimax
from data import load_data, get_line_chart_data
import plotly.express as px
import plotly.graph_objects as go

from data import load_data
from model_arima import load_model_arima1, load_model_arima2
import plotly.express as px
import plotly.graph_objects as go



st.markdown('''
    # Inflation Prediction
    ## A time series analysis project
''')

list_models = ['LSTM', 'SARIMAX', 'ARIMAX']
model_selection = st.selectbox('Select a model', list_models)

@st.cache
def predict(model_selection):

    # print is visible in the server output, not in the page
    print('Prediction made!')

    if model_selection == 'LSTM':
        model = load_model_lstm()
        results = prediction_lstm(model)

    elif model_selection == 'SARIMAX':
        path2 = 'data/final_df.csv'
        df = load_data(path2)
        y=df['RPI']
        y_test5 = y[163:164]
        model = load_model_sarimax()
        prediction = model.predict(start = 60)
        results = pd.DataFrame(columns = ['test_predictions', 'test_actual'], index = y_test5.index)
        results['test_predictions'] = prediction[163]
        results['test_actual'] = 345.2

    elif model_selection == 'ARIMAX':
        path3 = 'data/final_df.csv'
        df = load_data(path3)
        y = df['RPI']
        y_test5 = y[163:164]
        model1 = load_model_arima1()
        model2 = load_model_arima2()
        #arratst is the exogenous features for 2022-2023
        arratst = np.array([[343.54180382,  11.04269294, -10.64054131,  58.34667398]])
        prediction1 = model1.predict(start=61,end=61,exog = arratst,dynamic=True)
        prediction2 = model2.predict(start=61,end=61,exog = arratst,dynamic=True)
        prediction_value_1 = (prediction1.iloc[0]/10)*307.4
        prediction_value_2 = (prediction2.iloc[0]/10)*307.4
        prediction = (prediction_value_1+prediction_value_2)/2
        results = pd.DataFrame(columns = ['test_predictions', 'test_actual'], index = y_test5.index)
        results['test_predictions'] = prediction[163]
        results['test_actual'] = 345.2

    else:
        pass

    return results

##TO DO: FORMAT OUTPUT, round pred
##Add truth value
##Add chart comparing both



def plotly_lstm():
    df = get_line_chart_data().set_index('Date')
    model6 = load_model_lstm()
    y = df['RPI']
    test_results = prediction_lstm(model6)
    # test_results5.index = pd.to_datetime(test_results5.index)
    fig1 = px.line(y)
    fig2 = px.scatter(test_results["test_predictions"], color_discrete_sequence=['red'])
    fig3 = go.Figure(data=fig1.data + fig2.data)
    return fig3


def plotly_sarimax():
    df = get_line_chart_data().set_index('Date')
    model_sar = load_model_sarimax()
    y = df['RPI']
    test_results = prediction
    fig1 = px.line(y)
    fig2 = px.scatter(test_results["test_predictions"], color_discrete_sequence=['red'])
    fig_sar = go.Figure(data=fig1.data + fig2.data)
    return fig_sar

if st.button('Predict!'):
    prediction = predict(model_selection)
    st.markdown('## Prediction made!📈')
    st.dataframe(prediction.style.format("{:.2f}"))

    if model_selection == 'LSTM':
        fig_lstm = plotly_lstm()
        st.plotly_chart(fig_lstm)

    elif model_selection == 'SARIMAX':
        fig_sar = plotly_sarimax()
        st.plotly_chart(fig_sar)









    st.pyplot(plotly_lstm())
# st.plotly_chart(plotly_lstm())



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
