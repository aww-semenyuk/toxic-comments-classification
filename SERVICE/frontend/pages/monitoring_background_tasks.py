import streamlit as st
from utils_func.process_data import map_background_tasks
from logger_config import get_logger

logger = get_logger()

st.title("Мониторинг фоновых задач")

df = map_background_tasks()

if df.empty is False:
    st.table(df)
else:
    st.info("Нет активных задач в фоновом режиме.")
