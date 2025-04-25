import pandas as pd
import streamlit as st
import zipfile
import io
import hashlib
import time

from logger_config import get_logger


logger = get_logger()


@st.cache_data
def load_data(UploadedFile):
    logger.info(f"Successfuly uploaded {UploadedFile.name}")
    return pd.read_csv(UploadedFile)


def check_input_data(df):
    return {"toxic", "comment_text"}.issubset(df.columns)


def generate_random_hash():
    current_time = str(time.time())

    hash_object = hashlib.sha256(current_time.encode())
    hash_id = hash_object.hexdigest()[:8]

    return hash_id


def create_zip_from_csv(uploaded_file, zip_filename):
    zip_buffer = io.BytesIO()

    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        csv_content = uploaded_file.getvalue()
        zip_file.writestr(zip_filename, csv_content)

    return zip_buffer.getvalue()


def escape_quotes(text):
    return text.replace('"', '\\"').replace("'", "\\'")
