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
    model = st.selectbox("Выберите модель для обучение", ['Logistic Regression', 'SVC'])

    if model == 'Logistic Regression':
        penalty = st.selectbox("Penalty (Регуляризация)", ['l1', 'l2', 'elasticnet', 'none'])
        C = st.slider("C (Обратная сила регуляризации)", min_value=0.01, max_value=100.0, value=1.0, step=0.1)
        solver = st.selectbox("Solver (Оптимизация)", ['liblinear', 'lbfgs', 'saga'])
        max_iter = st.slider("Max Iter (Итерации)", min_value=100, max_value=1000, value=100, step=50)

    if model == 'SVC':
        C = st.slider("C (Регуляризация)", min_value=0.01, max_value=100.0, value=1.0, step=0.1)
        penalty = st.selectbox("Penalty (Регуляризация)", ['l2', 'l1'])
        loss = st.selectbox("Loss (Функция потерь)", ['squared_hinge', 'hinge'])
        dual = st.checkbox("Решение двойственной задачи (Dual)", value=True) if penalty == 'l2' else False
        class_weight = st.selectbox("Class Weight (Вес классов)", [None, 'balanced'])
        max_iter = st.slider("Max Iter (Максимум итераций)", min_value=100, max_value=5000, value=1000, step=100)


if model:
        pressed = st.button('Обучить модель')
        if pressed:
            if model == 'Logistic Regression':
                r_2_score, accuracy = learn_logistic_regression(shared_data, penalty, C, solver, max_iter)
                st.write(f'R2-score: {r_2_score}')
                st.write(f'Accuracy: {accuracy:.2f}')
                st.success('Модель обучена.')

            if model == 'SVC':
                r_2_score, accuracy = learn_LinearSVC_regression(shared_data, C, penalty, loss, dual, class_weight, max_iter)
                st.write(f'R2-score: {r_2_score}')
                st.write(f'Accuracy: {accuracy:.2f}')
                st.success('Модель обучена.')

else:
    st.write('Для обучения модели необходимо загрузить данные во вкладке "Main"')