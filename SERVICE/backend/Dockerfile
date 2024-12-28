FROM python:3.11-slim

WORKDIR /app
COPY . .

RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential gcc g++ libffi-dev libssl-dev && \
    rm -rf /var/lib/apt/lists/*

RUN pip install -r requirements.txt --no-cache-dir && \
    python -m spacy download en_core_web_sm && \
    python -m nltk.downloader stopwords

RUN apt-get purge -y --auto-remove build-essential gcc g++ && \
    apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/*

CMD ["python", "main.py"]