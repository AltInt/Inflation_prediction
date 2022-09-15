from multiprocessing import parent_process
from pyexpat import model
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from model_arima import load_model_arima1
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
        results['test_predictions'] = prediction
        results['test_actual'] = 345.2

    else:
        st.write('No model was chosen. Please refresh the page and try again.')
        return None

    return results

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


def plotly_arimax():
    df = get_line_chart_data().set_index('Date')
    model_ar1 = load_model_arima1()
    model_ar2 = load_model_arima2()
    arratst = np.array([[343.54180382,  11.04269294, -10.64054131,  58.34667398]])
    y = df['RPI']
    prediction1 = model_ar1.predict(start=61,end=61,exog = arratst,dynamic=True)
    prediction2 = model_ar2.predict(start=61,end=61,exog = arratst,dynamic=True)
    prediction_value_1 = ((prediction1.iloc[0]*0.01)+1)*307.4
    prediction_value_2 = ((prediction2.iloc[0]*0.01)+1)*307.4
    prediction = (prediction_value_1+prediction_value_2)/2

    test_result = pd.DataFrame(columns=['test_predictions', 'test_actual'])
    test_result['test_predictions'] = prediction
    test_result['test_actual'] = 345.2
    test_result['date'] = y.index[163]

    fig1 = px.line(y)
    fig2 = px.scatter(test_result["test_predictions"], color_discrete_sequence=['red'])
    fig_ar = go.Figure(data=fig1.data + fig2.data)
    return fig_ar



if st.button('Predict!'):
    st.balloons()
    prediction = predict(model_selection)


    if model_selection == 'LSTM':
        st.markdown('## Prediction made!📈')



        col1, col2, col3 = st.columns(3)
        col1.metric('Date', prediction.index[0])
        col2.metric('Prediction', prediction['test_predictions'], "+12.0%")
        col3.metric('Truth', prediction['test_actual'], "+12.3%")


        st.markdown('## ')
        st.markdown('### Results Dataframe')
        st.dataframe(prediction.style.format("{:.2f}"))

        fig_lstm = plotly_lstm()
        st.plotly_chart(fig_lstm)



    elif model_selection == 'SARIMAX':
        st.markdown('## Prediction made!📈')

        prediction['test_predictions'] = prediction['test_predictions'].astype(float).round(2)

        col1, col2, col3 = st.columns(3)
        col1.metric('Date', prediction.index[0])
        col2.metric('Prediction', prediction['test_predictions'], "+12.5%")
        col3.metric('Truth', prediction['test_actual'], "+12.3%")


        st.markdown('## ')
        st.markdown('### Results Dataframe')
        st.dataframe(prediction.style.format("{:.2f}"))




        fig_sar = plotly_sarimax()
        st.plotly_chart(fig_sar)






    elif model_selection == 'ARIMAX':
        fig_ar = plotly_arimax()
        st.plotly_chart(fig_ar)
