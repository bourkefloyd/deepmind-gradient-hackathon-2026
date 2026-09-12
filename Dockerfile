FROM python:3.12-slim

WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Word lists are not committed (data/README.md); fetch at build time.
RUN mkdir -p data \
 && curl -sL -o data/enable1.txt https://raw.githubusercontent.com/dolph/dictionary/master/enable1.txt \
 && curl -sL https://raw.githubusercontent.com/arstgit/high-frequency-vocabulary/master/30k.txt | tr -d '\r\t' | awk 'NF' > data/common-30k.txt \
 && wc -l data/enable1.txt data/common-30k.txt

COPY wordhunt ./wordhunt
COPY static ./static
COPY data/*.json ./data/

ENV PORT=8080
EXPOSE 8080
CMD exec uvicorn wordhunt.server:app --host 0.0.0.0 --port ${PORT} --ws-ping-interval 20 --ws-ping-timeout 20
