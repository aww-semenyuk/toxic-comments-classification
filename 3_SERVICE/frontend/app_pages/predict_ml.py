import streamlit as st
from logger_config import get_logger
import pandas as pd

from utils.client import RequestHandler
from utils.data_processing import escape_quotes

logger = get_logger()

handler = RequestHandler(logger)

st.subheader("Predict toxicity with Machine Learning")

models_resp = handler.get_models(is_dl=False)

if models_resp["is_success"]:
    tmp_df = pd.DataFrame(models_resp["response"].json())
    if tmp_df.empty:
        st.info("No models to predict with, train first")
    else:
        df_models = tmp_df
else:
    st.error(models_resp["response"].json()["detail"])

if "df_models" in locals():
    selected_model = st.selectbox(
        r"$\text{Select a model}$",
        df_models["name"].unique()
    )

    if 'text_areas' not in st.session_state:
        st.session_state.text_areas = [""]

    texts = []
    for i, text in enumerate(st.session_state.text_areas):
        new_text = st.text_area(
            f"Text {i + 1}",
            placeholder="Enter texts to predict toxicity for",
            value=text,
            key=f"text_area_{i}"
        )
        texts.append(new_text)

    if st.button("Add new textarea"):
        st.session_state.text_areas.append("")
        st.rerun()

    st.session_state.text_areas = texts

    X = [escape_quotes(t) for t in texts if t.strip() != ""]

    pressed_predict = st.button("Obtain predictions")
    if pressed_predict:
        logger.info(f"Texts added {texts}")
        logger.info(
            f"st.session_state.text_areas {st.session_state.text_areas}"
        )
        if not df_models[df_models["name"] == selected_model]["is_loaded"] \
                .values[0]:
            st.error("""The model is unloaded, \
                     load the model first and try again""")
        else:
            pred_resp = handler.predict(selected_model, X)
            if pred_resp["is_success"]:
                preds = pred_resp["response"] \
                    .json()["predictions"]
                preds_text = ["Toxic" if pred > 0 else "Not toxic"
                              for pred in preds]
                st.markdown(r'$\text{Predictions}$')
                st.table({"Text": X, "Predictions": preds_text})
            else:
                st.error(pred_resp["response"].json()["detail"])
