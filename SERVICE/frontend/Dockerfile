FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip install -r requirements.txt --no-cache-dir && \
    python -m nltk.downloader stopwords punkt punkt_tab wordnet \
    averaged_perceptron_tagger averaged_perceptron_tagger_eng

COPY . .

CMD ["streamlit", "run", "main.py", "--server.port=8501"]
