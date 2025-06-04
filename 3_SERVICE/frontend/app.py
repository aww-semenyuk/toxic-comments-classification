import streamlit as st

home_page = st.Page(
    'app_pages/file_upload.py',
    title='Homepage/data upload',
    icon=":material/upload_file:",
    default=True
)
eda_page = st.Page(
    'app_pages/eda.py',
    title='EDA',
    icon=":material/equalizer:"
)
train_page = st.Page(
    'app_pages/train_models.py',
    title="Train models",
    icon=":material/cycle:"
)
manage_page = st.Page(
    'app_pages/manage_models.py',
    title="Manage models",
    icon=":material/settings:"
)
bg_page = st.Page(
    'app_pages/bg_tasks.py',
    title="Monitor tasks",
    icon=":material/monitoring:"
)
predict_ml_page = st.Page(
    'app_pages/predict_ml.py',
    title="Predict with ML",
    icon=":material/network_node:"
)
predict_dl_page = st.Page(
    'app_pages/predict_dl.py',
    title="Predict with DL",
    icon=":material/network_node:"
)
scores_page = st.Page(
    'app_pages/predict_scores.py',
    title="Compare models by ROC",
    icon=":material/compare_arrows:"
)

pg = st.navigation({
    'Home': [home_page],
    'Models': [
        manage_page,
        train_page,
        predict_ml_page,
        predict_dl_page,
        scores_page
    ],
    'Analytics': [eda_page],
    'Tools': [bg_page]
}, expanded=True)

pg.run()
