import streamlit as st
import pandas as pd

# Данные для таблицы
data = {
    "Задача": ["Обработка пайпа", "Создание модели"],
    "Статус": ["В процессе", "Готово"]
}

# Создаем DataFrame
df = pd.DataFrame(data)

# Отображаем таблицу
st.table(df)