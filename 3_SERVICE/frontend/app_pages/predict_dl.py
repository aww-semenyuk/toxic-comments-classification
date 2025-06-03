import streamlit as st
from logger_config import get_logger
import pandas as pd

from utils.client import RequestHandler
from utils.data_processing import escape_quotes

logger = get_logger()

handler = RequestHandler(logger)

st.subheader("Predict toxicity with Deep Learning")

models_resp = handler.get_models()

if models_resp["is_success"]:
    tmp_df = pd.DataFrame(models_resp["response"].json())
    if tmp_df.empty:
        st.info("No models to predict with, train first")
    else:
        df_models = tmp_df
else:
    st.error(models_resp["response"].json()["detail"])

if "df_models" in locals():
    selected_model = st.selectbox(r"$\text{Select a model}$",
                                  df_models["id"].unique())
    text_X = st.text_area(
        r"$\text{Enter new line separated texts to predict toxicity for}$")
    text_X = escape_quotes(text_X)
    X = text_X.split('\n')
    pressed_predict = st.button("Obtain predictions")
    if pressed_predict:
        if not df_models[df_models["id"] == selected_model]["is_loaded"] \
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
