import streamlit as st

main_page = st.Page('pages/Main.py', title='Homepage/data upload', icon=":material/upload_file:", default=True)
eda_page = st.Page('pages/EDA.py', title='EDA', icon=":material/equalizer:")
learning_page = st.Page('pages/learning_and_settings_hyperparams.py', title="Train models", icon=":material/cycle:")
manage_page = st.Page('pages/manage_current_models.py', title="Manage models", icon=":material/settings:")
bg_page = st.Page('pages/monitoring_background_tasks.py', title="Monitor tasks", icon=":material/monitoring:")
pred_page = st.Page('pages/predict_models.py', title="Predict", icon=":material/network_node:")
scores_page = st.Page('pages/predict_scores.py', title="Compare models by ROC", icon=":material/compare_arrows:")

pg = st.navigation({
    'Home': [main_page],
    'Models': [learning_page, manage_page, pred_page, scores_page],
    'Analytics': [eda_page],
    'Tools': [bg_page]
}, expanded=True)

pg.run()
