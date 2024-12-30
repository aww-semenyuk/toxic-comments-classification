import streamlit as st
import pandas as pd

from logger_config import get_logger
from utils.client import RequestHandler


logger = get_logger()
handler = RequestHandler(logger)


st.subheader("Trained models management")

models_resp = handler.get_models()

if models_resp["is_success"]:
    tmp_df = pd.DataFrame(models_resp["response"].json()) \
        .loc[lambda df: df["is_trained"]]
    if tmp_df.empty:
        st.info("No trained models")
    else:
        df_models = tmp_df
else:
    st.error(models_resp["response"].json()["detail"])

if "df_models" in locals():
    if st.button("Remove all models (except default)"):
        remove_all_resp = handler.remove_all_models()
        if remove_all_resp["is_success"]:
            st.success('All models except defaults removed')
        else:
            st.error(f'{remove_all_resp["response"].json()["detail"]}')

    col_names = ['Model ID', 'Model type', 'LOAD/UNLOAD', 'DELETE']
    col_widths = [0.36, 0.3, 0.17, 0.17]
    cols = dict(zip(col_names, st.columns(col_widths)))

    for name, col in cols.items():
        col.write(name)

    for _, row in df_models.iterrows():
        cols = dict(zip(col_names, st.columns(col_widths)))
        cols['Model ID'].write(row['id'])
        cols['Model type'].write(row['type'])

        if row['is_loaded']:
            unload_button = cols['LOAD/UNLOAD'].button(
                'Unload',
                key=f"button_unload_{row['id']}",
                use_container_width=True)
            if unload_button:
                unload_resp = handler.unload_model(row["id"])
                if unload_resp["is_success"]:
                    st.success(f'Model {row["id"]} unloaded')
                else:
                    st.error(f'{unload_resp["response"].json()["detail"]}')
        else:
            load_button = cols['LOAD/UNLOAD'].button(
                'Load',
                key=f"button_load_{row['id']}",
                use_container_width=True)
            if load_button:
                load_resp = handler.load_model(row["id"])
                if load_resp["is_success"]:
                    st.success(f'Model {row["id"]} loaded')
                else:
                    st.error(f'{load_resp["response"].json()["detail"]}')

        delete_button = cols['DELETE'].button(
            'Delete',
            key=f"button_delete_{row['id']}",
            use_container_width=True)
        if delete_button:
            delete_resp = handler.remove_model(row["id"])
            if delete_resp["is_success"]:
                st.success(f'Model {row["id"]} deleted')
            else:
                st.error(f'{delete_resp["response"].json()["detail"]}')
