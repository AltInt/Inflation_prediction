
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from model_lstm import load_model_lstm, prediction_lstm
from model_sarimax import load_model_sarimax
from data import load_data
import plotly.express as px
import plotly.graph_objects as go



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

    return results






##TO DO: FORMAT OUTPUT, round pred
##Add truth value
##Add chart comparing both










def get_line_chart_data():

    return pd.read_csv('data/final_df.csv')

def plotly_lstm():

    df = get_line_chart_data().set_index('Date')
    model6 = load_model_lstm()
    y = df['RPI']
    test_results5 = prediction_lstm(model6)
    # test_results5.index = pd.to_datetime(test_results5.index)


    fig1 = px.line(y)
    fig2 = px.scatter(test_results5["test_predictions"], color_continuous_scale='Inferno')
    fig3 = go.Figure(data=fig1.data + fig2.data)
    return fig3

if st.button('Predict!'):
    prediction = predict(model_selection)

    st.markdown('## Prediction made!📈')
    st.dataframe(prediction.style.format("{:.2f}"))
    st.pyplot(plotly_lstm())
    # st.write(prediction)

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
