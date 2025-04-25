import streamlit as st
import pandas as pd
from sklearn.metrics import roc_curve, auc
import plotly.express as px
from io import BytesIO

from logger_config import get_logger
from utils.client import RequestHandler
from utils.data_processing import (
    load_data,
    create_zip_from_csv,
    check_input_data
)

logger = get_logger()
handler = RequestHandler(logger)

st.subheader("Plot ROC curves")

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
    selected_models = st.multiselect(
        r'$\large\text{Select a model}$',
        df_models["id"].unique(),
        default=df_models["id"].unique()[0]
    )

    use_train_data = st.checkbox("Use previous (train) data", value=True)
    if not use_train_data:
        uploaded_file = st.file_uploader(
            r'$\large\text{Upload data to plot}$',
            type=['csv'],
            help='Data must contain columns named "comment_text" and "toxic"',
            label_visibility='visible'
        )
        if uploaded_file:
            data = load_data(uploaded_file)
            if not check_input_data(data):
                st.error("""Your data's format is incorrect, \
                         check that it contains columns \
                         "comment_text" and "toxic"
                        """)
            else:
                zip_data = create_zip_from_csv(uploaded_file,
                                               uploaded_file.name)
                st.session_state['zipped_csv_new'] = zip_data

    pressed_predict = st.button("Построить графики")
    if pressed_predict:
        st.divider()

        if use_train_data and 'zipped_csv' in st.session_state:
            zipped_csv = st.session_state['zipped_csv']
        elif not use_train_data and 'zipped_csv_new' in st.session_state:
            zipped_csv = st.session_state['zipped_csv_new']

        if 'zipped_csv' not in locals():
            st.error('Upload data first')
        else:
            scores_resp = handler.predict_scores(selected_models, zipped_csv)
            if not scores_resp['is_success']:
                st.error(scores_resp["response"].json()["detail"])
            else:
                resp_df = pd.read_csv(BytesIO(scores_resp['response'].content))

                all_data = []

                for model_id in resp_df['model_id'].unique():
                    data_ = resp_df[resp_df['model_id'] == model_id]
                    fpr, tpr, _ = roc_curve(data_['y_true'], data_['scores'])
                    auc_score = auc(fpr, tpr)
                    model_data = pd.DataFrame({
                        "False Positive Rate": fpr,
                        "True Positive Rate": tpr,
                        "Model": [f"{model_id} (AUC={auc_score:.2f})"] * len(fpr)
                    })
                    all_data.append(model_data)

                if all_data:
                    all_data_df = pd.concat(all_data, ignore_index=True)
                    fig = px.line(
                        all_data_df,
                        x="False Positive Rate",
                        y="True Positive Rate",
                        color="Model",
                        title="Receiver Operating Characteristic (ROC) Curve"
                    )
                    fig.update_layout(
                        xaxis_title="False Positive Rate",
                        yaxis_title="True Positive Rate",
                        legend_title="Models",
                        showlegend=True,
                        width=900,
                        height=600,
                    )
                    st.plotly_chart(fig, use_container_width=False)
                else:
                    st.warning("No models available for plotting.")
