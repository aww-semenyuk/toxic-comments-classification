FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .

RUN apt-get update && \
    apt-get upgrade && \
    pip install --upgrade pip && \
    pip install -r requirements.txt --no-cache-dir && \
    python -c "import nltk; nltk.download('stopwords'); \
                            nltk.download('punkt'); \
                            nltk.download('punkt_tab'); \
                            nltk.download('wordnet'); \
                            nltk.download('averaged_perceptron_tagger'); \
                            nltk.download('averaged_perceptron_tagger_eng');"

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "main.py", "--server.port=8501"]
