import streamlit as st

from logger_config import get_logger
from utils.client import RequestHandler


logger = get_logger()
handler = RequestHandler(logger)


st.subheader("Models Info")

models_resp = handler.get_models()

if models_resp["is_success"]:
    models = models_resp["response"].json()
    if not models:
        st.info("Models not found")
else:
    st.error(models_resp["response"].json()["detail"])

if "models" in locals():
    name_to_model = {model["name"]: model for model in models}
    model_names = list(name_to_model.keys())

    selected_model_name = st.selectbox(
        r"$\text{Select a model}$",
        options=model_names,
        index=0
    )
    selected_model = name_to_model[selected_model_name]

    st.subheader("General info")
    st.write(f"**Type:** `{selected_model['type']}`")
    st.write(f"**Is trained:** {'✅' if selected_model['is_trained'] else '❌'}")
    st.write(f"**Is loaded:** {'✅' if selected_model['is_loaded'] else '❌'}")

    st.subheader("Model params")
    st.json(selected_model["model_params"])

    st.subheader("Vectorizer params")
    st.json(selected_model["vectorizer_params"])

    if st.button("🔄 Update info"):
        st.cache_data.clear()
        st.rerun()
