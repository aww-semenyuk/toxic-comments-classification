FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .

RUN apt-get update && apt-get install -y && \
    pip install -r requirements.txt --no-cache-dir && \
    python -m nltk.downloader stopwords wordnet punkt averaged_perceptron_tagger

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "Main.py", "--server.port=8501"]
