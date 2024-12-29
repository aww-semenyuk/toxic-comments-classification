import streamlit as st
from utils_func.process_data import map_current_models, predict_scores_action
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

st.title("Получение кривых обучения")

df = map_current_models()

if not df.empty:

    model_ids = df["id"].unique()

    selected_models = st.multiselect(
        "Выберите id моделей",
        options=model_ids,
        format_func=lambda x: f"Модель id-{x}",
    )

    pressed_predict = st.button("Получить кривые обучения")

    if pressed_predict or "roc_curves_data" in st.session_state:
        if pressed_predict:
            if len(selected_models) > 0:
                res_scores_df = predict_scores_action(
                    selected_models, st.session_state['zipped_csv']
                )

                if ('y_true' in res_scores_df.columns
                        and 'scores' in res_scores_df.columns):

                    st.session_state["roc_curves_data"] = {
                        model_id: {
                            "y_true":
                                res_scores_df[
                                    res_scores_df['model_id'] == model_id
                                ]['y_true'],
                            "scores":
                                res_scores_df[
                                    res_scores_df['model_id'] == model_id
                                ]['scores']
                        }
                        for model_id in selected_models
                    }

                else:
                    st.error(
                        "Невозможно "
                        "построить ROC-кривую: "
                        "отсутствуют необходимые данные."
                    )
            else:
                st.error("Выберите хотя бы одну модель.")

        # Получаем сохраненные данные ROC-кривых
        if "roc_curves_data" in st.session_state:
            roc_curves_data = st.session_state["roc_curves_data"]

            # Словарь для управления видимостью моделей
            visibility = {}
            for model_id in roc_curves_data.keys():
                visibility[model_id] = st.checkbox(
                    f"Показать ROC для модели id-{model_id}", value=True
                )

            plt.figure(figsize=(8, 6))  # Настройка размера окна для графиков

            # Для каждой модели строим ROC кривую, если чекбокс включен
            for model_id, data in roc_curves_data.items():
                if visibility[model_id]:  # Проверяем состояние чекбокса
                    fpr, tpr, _ = roc_curve(data['y_true'], data['scores'])
                    roc_auc = auc(fpr, tpr)

                    # Визуализация ROC-кривой
                    plt.plot(
                        fpr,
                        tpr,
                        lw=2,
                        label=f'Model id-{model_id} (AUC = {roc_auc:.2f})'
                    )

            # Общие элементы графика
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic (ROC) Curve')
            plt.legend(loc="lower right")
            plt.grid()

            # Отображаем график
            st.pyplot(plt)
else:
    st.info("Нет активных задач в фоновом режиме.")
