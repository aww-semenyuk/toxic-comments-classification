import streamlit as st
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from process_data import learn_logistic_regression, learn_LinearSVC_regression

st.header('Обучение модели')
shared_data = None
if 'shared_data' in st.session_state:
    shared_data = st.session_state['shared_data']
if shared_data is not None:
    model = st.selectbox("Выберите модель для обучение", ['Logistic Regression', 'SVC', 'Кастомная модель'])
    if model:
        pressed = st.button('Обучить модель')
        if pressed:
            if model == 'Logistic Regression':
                r_2_score = learn_logistic_regression(shared_data)
                st.write(f'R2-score: {r_2_score}')
                st.success('Модель обучена.')

            if model == 'SVC':
                r_2_score = learn_LinearSVC_regression(shared_data)
                st.write(f'R2-score: {r_2_score}')
                st.success('Модель обучена.')

else:
    st.write('Для обучения модели необходимо загрузить данные во вкладке "Main"')