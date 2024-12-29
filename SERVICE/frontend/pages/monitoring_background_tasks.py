# Мониторинг фоновых задач
import streamlit as st
from utils_func.process_data import map_background_tasks

st.set_page_config(
    page_title="Мониторинг фоновых задач",
)

st.title("Мониторинг фоновых задач")

df = map_background_tasks()

if df.empty is False:
    st.table(df)
else:
    st.info("Нет активных задач в фоновом режиме.")
