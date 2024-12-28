import streamlit as st
from process_data import map_current_models, delete_all_models, delete_action

st.title("Управление текущими моделями")

df = map_current_models()

if df.empty is False:
    header_col1, header_col2, header_col3, header_col4, header_col5 = st.columns([2, 3, 2, 2, 2])
    header_col1.write("**ID**")
    header_col2.write("**Тип модели**")
    header_col3.write("**Модель обучена**")
    header_col4.write("**Модель загружена**")
    header_col5.write("**Действие**")
    for i, row in df.iterrows():
        col1, col2, col3, col4, col5 = st.columns([2, 3, 2, 2, 2])
        col1.write(row["id"])
        col2.write(row["Тип модели"])
        col3.write(row["Модель обучена"])
        col4.write(row["Модель загружена"])
        pressed = col5.button("Выполнить", key=f"button_{row['id']}")
        if pressed:
            err = delete_action(row["id"])
            if err is not None:
                st.error(f"Ошибка при удалении модели {row['id']}: {err}")
            else:
                st.success(f"Модель удалена {row['id']}.")

    pressed = st.button("Удалить текущие модели")
    if pressed:
        delete_all_models()
        st.success("Текущие модели удалены.")
else:
    st.info("Нет активных задач в фоновом режиме.")
