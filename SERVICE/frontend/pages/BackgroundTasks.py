import streamlit as st
import pandas as pd
import asyncio
from process_data import map_background_tasks

# Streamlit приложение
st.title("Мониторинг фоновых задач")

tasks = map_background_tasks()

if tasks:
    st.subheader("Статус задач:")
    # df = pd.DataFrame(tasks)
    # st.table(df)
else:
    st.info("Нет активных задач в фоновом режиме.")
