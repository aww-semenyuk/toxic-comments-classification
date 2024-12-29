import streamlit as st
from process_data import map_current_models, predict_scores_action
from sklearn.metrics import roc_curve, auc

st.title("Получение кривых обучения")

df = map_current_models()

if df.empty is False:
    st.subheader("Выберите модель")
    selected_model = st.selectbox("Модель", df["id"].unique())
    pressed_predict = st.button("Получить кривые обучения")
    if pressed_predict is True:
        res_scores = predict_scores_action(selected_model, st.session_state['zipped_csv'])
        if res_scores is None:
            st.error("Ошибка при получении предсказания.")
        else:
            st.success(f"Предсказанное значение: {res_scores}")

        y_true = res_scores

        # # Вычисление ROC-кривой
        # fpr, tpr, thresholds = roc_curve(y_true, scores)
        # roc_auc = auc(fpr, tpr)
        #
        # # Визуализация ROC-кривой
        # plt.figure(figsize=(8, 6))
        # plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        # plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        # plt.xlabel('False Positive Rate')
        # plt.ylabel('True Positive Rate')
        # plt.title('Receiver Operating Characteristic (ROC) Curve')
        # plt.legend(loc="lower right")
        # plt.grid()
        # plt.show()


else:
    st.info("Нет активных задач в фоновом режиме.")
