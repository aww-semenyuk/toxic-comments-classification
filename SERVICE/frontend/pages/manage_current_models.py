import streamlit as st
from utils_func.process_data import (
    map_current_models,
    delete_all_models,
    delete_action,
    load_model_action,
    unload_model_action
)

st.title("Управление текущими моделями")

df = map_current_models()

if df.empty is False:
    pressed = st.button("Удалить все модели")
    if pressed:
        delete_all_models()
        st.success("Текущие модели удалены.")
    st.divider()
    (
        header_col1,
        header_col2,
        header_col3,
        header_col4,
        header_col5,
    ) = st.columns([2, 3, 2, 2, 2])

    header_col1.write("**ID**")
    header_col2.write("**Тип модели**")
    header_col3.write("**Модель обучена**")
    header_col4.write("**Загрузить/Выгрузить модель**")
    header_col5.write("**Действие**")
    for i, row in df.iterrows():
        col1, col2, col3, col4, col5 = st.columns([2, 3, 2, 2, 2])
        col1.write(row["id"])
        col2.write(row["Тип модели"])
        col3.write(row["Модель обучена"])
        if row["Модель загружена"] is True:
            pressed_unload = col4.button(
                "Выгрузить модель", key=f"button_unload_{row['id']}"
            )
            if pressed_unload:
                err = unload_model_action(row["id"])
                if err is not None:
                    st.error(f"Ошибка при выгрузки модели {row['id']}: {err}")
                else:
                    st.success(f"Модель выгружена {row['id']}.")
        else:
            pressed_load = col4.button(
                "Загрузить модель", key=f"button_load_{row['id']}"
            )
            if pressed_load:
                err = load_model_action(row["id"])
                if err is not None:
                    st.error(f"Ошибка при загрузки модели {row['id']}: {err}")
                else:
                    st.success(f"Модель загружена {row['id']}.")

        pressed = col5.button("Удалить модель", key=f"button_{row['id']}")
        if pressed:
            err = delete_action(row["id"])
            if err is not None:
                st.error(
                    f"Ошибка при удалении модели id: {row['id']} err: {err}"
                )
            else:
                st.success(f"Модель удалена {row['id']}.")

else:
    st.info("Нет активных задач в фоновом режиме.")
