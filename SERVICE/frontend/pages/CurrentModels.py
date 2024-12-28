import streamlit as st
from process_data import map_current_models, delete_all_models

st.title("Управление текущими моделями")

df = map_current_models()

if df.empty is False:
    st.table(df)
    pressed = st.button("Удалить текущие модели")
    if pressed:
        delete_all_models()
        st.success("Текущие модели удалены.")
else:
    st.info("Нет активных задач в фоновом режиме.")
