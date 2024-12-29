import streamlit as st
import plotly.graph_objects as go

from utils_func.process_data import (
    learn_logistic_regression,
    learn_LinearSVC_regression,
    learn_naive_bayes
)

st.header('Обучение модели')
zipped_csv = None
model = None

if 'results' not in st.session_state:
    st.session_state['results'] = {}

if 'zipped_csv' in st.session_state:
    zipped_csv = st.session_state['zipped_csv']

if zipped_csv is not None:
    model = st.selectbox(
        "Выберите модель для обучение",
        ['Logistic Regression', 'SVC', 'Naive Bayes']
    )

    vectorizer_type = st.selectbox(
        "Выберите векторизацию текста",
        ['tf_idf', 'bag_of_words']
    )

    if model == 'Logistic Regression':
        penalty = st.selectbox(
            "Penalty (Регуляризация)", ['l1', 'l2', 'elasticnet', 'none']
        )
        C = st.slider(
            "C (Обратная сила регуляризации)",
            min_value=0.01,
            max_value=100.0,
            value=1.0,
            step=0.1
        )
        solver = st.selectbox(
            "Solver (Оптимизация)", ['liblinear', 'lbfgs', 'saga']
        )
        max_iter = st.slider(
            "Max Iter (Итерации)",
            min_value=100,
            max_value=1000,
            value=100,
            step=50
        )

    if model == 'SVC':
        C = st.slider(
            "C (Регуляризация)",
            min_value=0.01,
            max_value=100.0,
            value=1.0,
            step=0.1
        )
        penalty = st.selectbox("Penalty (Регуляризация)", ['l2', 'l1'])
        loss = st.selectbox(
            "Loss (Функция потерь)", ['squared_hinge', 'hinge']
        )
        dual = (
            st.checkbox("Решение двойственной задачи (Dual)", value=True)
            if penalty == "l2"
            else False
        )
        class_weight = st.selectbox(
            "Class Weight (Вес классов)",
            [None, 'balanced']
        )
        max_iter = st.slider(
            "Max Iter (Максимум итераций)",
            min_value=100,
            max_value=5000,
            value=1000,
            step=100
        )

    if model == 'Naive Bayes':
        alpha = st.slider(
            "Alpha (Сглаживание)",
            min_value=0.01,
            max_value=10.0,
            value=1.0,
            step=0.1
        )
        fit_prior = st.checkbox(
            "Fit Prior (Использовать априорные вероятности)",
            value=True
        )

if model:
    pressed = st.button('Обучить модель')
    if pressed:
        if model == 'Logistic Regression':
            err = learn_logistic_regression(
                data=zipped_csv,
                penalty=penalty,
                C=C,
                solver=solver,
                max_iter=max_iter,
                vectorizer_type=vectorizer_type,
            )
            if err is None:
                st.success('Модель обучена.')
            else:
                st.error('Ошибка при обучении модели.')

        if model == 'SVC':
            err = learn_LinearSVC_regression(
                data=zipped_csv,
                C=C,
                penalty=penalty,
                loss=loss,
                dual=dual,
                class_weight=class_weight,
                max_iter=max_iter,
                vectorizer_type=vectorizer_type
            )
            if err is None:
                st.success('Модель обучена.')
            else:
                st.error('Ошибка при обучении модели.')

        if model == 'Naive Bayes':
            err = learn_naive_bayes(
                data=zipped_csv,
                alpha=alpha,
                fit_prior=fit_prior,
                vectorizer_type=vectorizer_type
            )
            if err is None:
                st.success('Модель обучена.')
            else:
                st.error('Ошибка при обучении модели.')

else:
    st.write(
        'Для обучения модели необходимо загрузить данные в главной вкладке.'
    )

# Отображение общего графика
if st.session_state['results']:
    st.subheader("Результаты моделей")
    st.write("Выберите метрики и модели для отображения:")

    # Чекбоксы для метрик
    show_f1 = st.checkbox("Показывать F1-score", value=True)
    show_accuracy = st.checkbox("Показывать Accuracy", value=True)

    # Чекбоксы для моделей
    show_models = {
        model_name: st.checkbox(f"Показывать {model_name}", value=True)
        for model_name in st.session_state['results'].keys()
    }

    fig = go.Figure()

    for model_name, metrics in st.session_state['results'].items():
        if show_models.get(model_name, False):
            if show_f1 and 'F1-score' in metrics:
                fig.add_trace(
                    go.Bar(
                        name=f"{model_name} - F1-score",
                        x=[model_name],
                        y=[metrics['F1-score']]
                    )
                )
            if show_accuracy and 'Accuracy' in metrics:
                fig.add_trace(
                    go.Bar(
                        name=f"{model_name} - Accuracy",
                        x=[model_name],
                        y=[metrics['Accuracy']]
                    )
                )

    fig.update_layout(
        title="Метрики моделей",
        barmode='group',
        xaxis_title="Модели",
        yaxis_title="Значение метрик",
        legend_title="Метрики"
    )
    st.plotly_chart(fig)
