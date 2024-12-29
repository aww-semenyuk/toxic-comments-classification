FROM python:3.11-slim

WORKDIR /app
COPY . .

RUN pip install -r requirements.txt --no-cache-dir && \
    python -m spacy download en_core_web_sm && \
    python -m nltk.downloader stopwords

CMD ["python", "main.py"]
