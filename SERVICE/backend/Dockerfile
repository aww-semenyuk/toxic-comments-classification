FROM python:3.11-slim

WORKDIR /app
COPY . .

RUN apt-get update && \
    apt-get upgrade && \
    apt-get install --no-install-recommends -y curl && \
    pip install --upgrade pip && \
    pip install -r requirements.txt --no-cache-dir && \
    python -m spacy download en_core_web_sm && \
    python -m nltk.downloader stopwords && \
    rm -rf /var/lib/apt/lists/*

CMD ["python", "main.py"]
