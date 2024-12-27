import streamlit as st
import pandas as pd
import plotly.express as px
from time import time

import nltk
import string
import re
import unicodedata

nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("punkt")

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import MaxAbsScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, precision_recall_curve, roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split


# Функция для получения данных PR-кривой
def get_pr_curve_data(model, model_id, model_name, X_test, y_test):
    try:
        if hasattr(model, "decision_function"):
            y_scores = model.decision_function(X_test)
        else:
            y_scores = model.predict_proba(X_test)[:, 1]

        precision, recall, _ = precision_recall_curve(y_test, y_scores)
        return pd.DataFrame({
            "Precision": precision,
            "Recall": recall,
            "Model": f"{model_name} (ID={model_id})",
            "Model ID": model_id
        })
    except Exception as e:
        st.error(f"Error generating PR curve data for model {model_name}: {e}")
        return pd.DataFrame()


# Функция для получения данных ROC-кривой
def get_roc_curve_data(model, model_id, model_name, X_test, y_test):
    try:
        if hasattr(model, "decision_function"):
            y_scores = model.decision_function(X_test)
        else:
            y_scores = model.predict_proba(X_test)[:, 1]

        fpr, tpr, _ = roc_curve(y_test, y_scores)
        auc = roc_auc_score(y_test, y_scores)
        return pd.DataFrame({
            "False Positive Rate": fpr,
            "True Positive Rate": tpr,
            "Model": f"{model_name} (ID={model_id})",
            "Model ID": model_id,
            "AUC": [auc] * len(fpr)
        })
    except Exception as e:
        st.error(f"Error generating ROC curve data for model {model_name}: {e}")
        return pd.DataFrame()



st.set_page_config(layout="wide")
st.title("Classification of Toxic Comments Service")

if "pr_data" not in st.session_state:
    st.session_state.pr_data = pd.DataFrame()

if "roc_data" not in st.session_state:
    st.session_state.roc_data = pd.DataFrame()

if "previous_models" not in st.session_state:
    st.session_state.previous_models = []

if "model_counter" not in st.session_state:
    st.session_state.model_counter = 1


# Класс для предобработки текста
class PreprocessLemmaTokenizer:
    def __init__(self):
        self.wnl = WordNetLemmatizer()
        self.tknzr = TweetTokenizer()
        self.stopwords = set(stopwords.words("english"))
        self.punctuation = set(string.punctuation)

        self.url_re = re.compile(r"(https?://\S+|www\.\S+)", re.IGNORECASE)
        self.symbols_re = re.compile(r"[^a-zA-Zа-яА-Я0-9\s]")
        self.spaces_re = re.compile(r"\s+")

    @staticmethod
    def _normalize(doc):
        return unicodedata.normalize("NFKD", doc).encode("ascii", "ignore").decode("utf-8", "ignore")

    def _clean_txt(self, text):
        c_t = self.url_re.sub("", text.lower())
        c_t = self.symbols_re.sub(" ", c_t)
        c_text = self.spaces_re.sub(" ", c_t).strip()
        return c_text

    def __call__(self, doc):
        clean_doc = self._normalize(self._clean_txt(doc))
        tokens = self.tknzr.tokenize(clean_doc)
        return [
            self.wnl.lemmatize(token)
            for token in tokens
            if token.lower() not in self.stopwords and token not in self.punctuation
        ]


# Словарь доступных моделей
models = {
    "Logistic Regression": LogisticRegression,
    "Linear SVC": LinearSVC,
    "Naive Bayes": MultinomialNB
}


# Загрузка данных
st.header("Upload Data")
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file:
    try:
        data = pd.read_csv(uploaded_file, index_col=0)

        if "comment_text" not in data.columns or "toxic" not in data.columns:
            st.warning("Dataset must have 'comment_text' and 'toxic' columns.")
        else:
            

            data["tokens_ws"] = data["comment_text"].apply(PreprocessLemmaTokenizer())
            data["ctws"] = data["tokens_ws"].apply(lambda x: " ".join(x))

            # Выбор модели
            st.header("Model Training")
            model_name = st.selectbox("Select a model:", list(models.keys()))

            # Гиперпараметры
            if model_name == "Logistic Regression":
                c = st.slider("C (Regularization)", 0.01, 10.0, 1.0)
                max_iter = st.number_input("Max Iterations", 100, 5000, 1000, step=100)
                model_params = {"C": c, "max_iter": max_iter}

            elif model_name == "Linear SVC":
                c = st.slider("C (Regularization)", 0.01, 10.0, 1.0)
                max_iter = st.number_input("Max Iterations", 100, 5000, 1000, step=100)
                model_params = {"C": c, "max_iter": max_iter}

            elif model_name == "Naive Bayes":
                alpha = st.slider("Alpha (Smoothing)", 0.01, 10.0, 1.0)
                model_params = {"alpha": alpha}

            if st.button("Train Model"):
                model_id = st.session_state.model_counter
                st.session_state.model_counter += 1

                X_train, X_test, y_train, y_test = train_test_split(
                    data["ctws"], data["toxic"], test_size=0.2, random_state=42
                )

                pipeline = Pipeline([
                    ("vectorizer", CountVectorizer()),
                    ("scaler", MaxAbsScaler()),
                    ("classifier", models[model_name](**model_params))
                ])

                pipeline.fit(X_train, y_train)

                preds = pipeline.predict(X_test)

                report_df = pd.DataFrame(classification_report(y_test, preds, output_dict=True)).transpose()

                st.session_state.previous_models.append({
                    "id": model_id,
                    "name": model_name,
                    "report": report_df
                })

                # PR-кривая
                pr_data = get_pr_curve_data(pipeline, model_id, model_name, X_test, y_test)
                st.session_state.pr_data = pd.concat([st.session_state.pr_data, pr_data], ignore_index=True)

                # ROC-кривая
                roc_data = get_roc_curve_data(pipeline, model_id, model_name, X_test, y_test)
                st.session_state.roc_data = pd.concat([st.session_state.roc_data, roc_data], ignore_index=True)

                st.success(f"Model {model_name} (ID={model_id}) trained successfully!")

            # Очистка сохранённых данных
            if st.button("Clear All Saved Data"):
                st.session_state.pr_data = pd.DataFrame()
                st.session_state.roc_data = pd.DataFrame()
                st.session_state.previous_models = []
                st.session_state.model_counter = 1
                st.write("All saved data has been cleared.")

            # Отчёты
            st.header("View Model Reports")
            if st.session_state.previous_models:
                available_ids = [m["id"] for m in st.session_state.previous_models]

                selected_model_id = st.selectbox("Select Model ID to view report:", available_ids)

                selected_model = next(
                    (m for m in st.session_state.previous_models if m["id"] == selected_model_id),
                    None
                )

                if selected_model:
                    st.subheader(f"Report for Model ID {selected_model_id} ({selected_model['name']})")
                    st.dataframe(selected_model["report"])
                else:
                    st.error("Model not found.")
            else:
                st.write("No models have been trained yet.")

            # Графики PR и ROC
            st.header("Graphs")

            if not st.session_state.pr_data.empty:
                pr_fig = px.line(
                    st.session_state.pr_data,
                    x="Recall",
                    y="Precision",
                    color="Model",
                    line_group="Model ID",
                    title="PR Curves",
                    width=1200,
                    height=800
                )
                st.plotly_chart(pr_fig)

            if not st.session_state.roc_data.empty:
                roc_fig = px.line(
                    st.session_state.roc_data,
                    x="False Positive Rate",
                    y="True Positive Rate",
                    color="Model",
                    line_group="Model ID",
                    title="ROC Curves",
                    width=1200,
                    height=800
                )
                st.plotly_chart(roc_fig)

    except Exception as e:
        st.error(f"Error loading file: {e}")
else:
    st.write("Please upload a CSV file.")