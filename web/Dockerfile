FROM python:3.11-slim

RUN apt-get update && apt-get install -y openssl && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Generate self-signed cert for HTTPS (required for browser mic access)
RUN openssl req -x509 -newkey rsa:2048 -keyout /app/key.pem -out /app/cert.pem \
    -days 3650 -nodes -subj "/CN=speech-translator"

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8089", "--ssl-keyfile", "/app/key.pem", "--ssl-certfile", "/app/cert.pem"]
