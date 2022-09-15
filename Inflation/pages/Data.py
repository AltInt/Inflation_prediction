import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def get_line_chart_data():

    return pd.read_csv('data/final_df.csv')



df = get_line_chart_data().set_index('Date')


def get_dataframe_data():



    return df


df = get_dataframe_data()
st.markdown('# Data')
st.text('RPI, CPI and 12 exogenous features, 2009 - Present')
st.write(df)

def corr_matrix():
    fig, ax = plt.subplots()
    df =  get_line_chart_data()
    corr = df.corr()
    sns.heatmap(corr, ax=ax)

    return fig

st.markdown('# Correlation Matrix')
st.text('Heatmap that illustrates the correlation between all variables')
st.pyplot(corr_matrix())

def area_graph():
    df = get_line_chart_data()
    rpi_yoy_df = df[['RPI YOY', 'Date']]
    rpi_yoy_df = rpi_yoy_df.set_index('Date')
    return rpi_yoy_df


st.markdown('# YoY RPI')
st.text('Chart representing year on year inflation, 2009-Present')



rpi_yoy_df = area_graph()
st.area_chart(rpi_yoy_df)
