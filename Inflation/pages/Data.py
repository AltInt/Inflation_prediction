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
st.write(df)

def corr_matrix():
    fig, ax = plt.subplots()
    df =  get_line_chart_data()
    corr = df.corr()
    sns.heatmap(corr, ax=ax)

    return fig

st.pyplot(corr_matrix())
