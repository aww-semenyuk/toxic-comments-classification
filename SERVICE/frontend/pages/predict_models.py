import streamlit as st
from utils_func.process_data import map_current_models, predict_action
from logger_config import get_logger

logger = get_logger()

st.title("Получение предсказанных значение и оценка")

df = map_current_models()

if not df.empty:
    st.subheader("Выберите модель")
    selected_model = st.selectbox("Модель", df["id"].unique())
    text_X = st.text_area("Введите текст для предсказания")
    pressed_predict = st.button("Получить предсказание")
    if pressed_predict is True:
        res = predict_action(selected_model, text_X)
        if res is None:
            st.error("Ошибка при получении предсказания.")
        else:
            st.success(f"Предсказанное значение: {res}")


else:
    st.info("Нет моделей для предсказания.")
