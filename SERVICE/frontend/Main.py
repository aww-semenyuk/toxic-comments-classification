import streamlit as st
import pandas as pd
from utils_func.process_data import (
    is_data_correct,
    logging,
    create_zip_from_csv,
)
from utils_func.setup_logging import setup_logging

st.title('Приложение для анализа и визуализации данных о токсичных комментариев')
st.header('Загрузите файл с комментариями')


@st.cache_data
def load_data(filepath):
    logging.info(f"Данные с файла {filepath} загружены!")
    return pd.read_csv(filepath)


# Запуск настроек логирования
setup_logging()

uploaded_file = st.file_uploader('Загрузите файл файл с комментариями', type=['csv'])

if uploaded_file is not None:
    data = load_data(uploaded_file)
    if is_data_correct(data) is False:
        st.error(
            'Необходимые столбцы: toxic и comment_text - отсутствуют в вашем файле.'
        )
    else:
        st.dataframe(data)

        st.session_state['shared_data'] = data
        zip_data = create_zip_from_csv(uploaded_file, uploaded_file.name)

        st.session_state['zipped_csv'] = zip_data


else:
    st.write('Файл не выбран.')
