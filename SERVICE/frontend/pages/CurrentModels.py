import streamlit as st
import pandas as pd
from process_data import map_current_models

st.title("Управление текущими моделями")

df = map_current_models()

if df.empty is False:
    st.table(df)
else:
    st.info("Нет активных задач в фоновом режиме.")
