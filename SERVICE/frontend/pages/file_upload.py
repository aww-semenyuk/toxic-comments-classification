import streamlit as st
import pandas as pd

from logger_config import get_logger
from utils_func.process_data import (
    is_data_correct,
    create_zip_from_csv
)

logger = get_logger()

st.title("Toxic comments classification app")
st.header("""
App for analyzing data, \
managing models and \
obtaining predicitions \
for the task of classificating toxic comments
""", divider='gray')


@st.cache_data
def load_data(filepath):
    """File data loader."""
    logger.info("Данные с файла %s загружены!", filepath)
    return pd.read_csv(filepath)


uploaded_file = st.file_uploader(
    r'$\large\text{Upload data to analyze/train model with}$',
    type=['csv'],
    help='data must contain columns named "comment_text" and "toxic"',
    label_visibility='visible'
)

if uploaded_file:
    data = load_data(uploaded_file)
    if not is_data_correct(data):
        st.error("""Your data's format is incorrect, \
                 check that is contains columns "comment_text" and "toxic"
                 """)
    else:
        st.markdown(r'$\text{Data preview}$')
        st.table(data[['comment_text', 'toxic']].head())

        zip_data = create_zip_from_csv(uploaded_file, uploaded_file.name)

        st.session_state['shared_data'] = data
        st.session_state['zipped_csv'] = zip_data
