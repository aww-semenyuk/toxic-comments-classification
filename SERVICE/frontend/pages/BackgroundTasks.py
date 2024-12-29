import streamlit as st
from process_data import map_background_tasks

st.title("Мониторинг фоновых задач")

df = map_background_tasks()

if df.empty is False:
    st.table(df)
else:
    st.info("Нет активных задач в фоновом режиме.")
