import streamlit as st

from logger_config import get_logger
from utils.client import RequestHandler
from utils.streamlit_helpers import (
    select_model_parameters,
    select_vectorizer_parameters)
from utils.data_processing import generate_random_hash

logger = get_logger()
handler = RequestHandler(logger)

st.subheader('Model training')


if 'zipped_csv' not in st.session_state:
    st.info('Upload data first')
else:
    zipped_csv = st.session_state['zipped_csv']

    select_cols = st.columns([0.5, 0.5])

    with select_cols[0]:
        model_choice, model_params = select_model_parameters()
    with select_cols[1]:
        vectorizer_choice, vectorizer_params = select_vectorizer_parameters()

    model_id = st.text_input(
        r"$\large\text{Enter the name of your model}$")
    if not model_id:
        model_id = f"{model_choice}_{generate_random_hash()}"

    if st.button("Apply params and fit"):

        train_resp = handler.train_model(
            zipped_csv,
            model_id,
            model_choice,
            model_params,
            vectorizer_choice,
            vectorizer_params)

        if train_resp["is_success"]:
            st.success(train_resp["response"].json()["message"])
        else:
            st.error(train_resp["response"].json()["detail"])
