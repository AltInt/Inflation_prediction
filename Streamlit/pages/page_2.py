import streamlit as st
import pandas as pd


def get_line_chart_data():

    return pd.read_csv('../data/final_df.csv')



df = get_line_chart_data().set_index('Date')


def get_dataframe_data():



    return df


df = get_dataframe_data()
st.write(df)
