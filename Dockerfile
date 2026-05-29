FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY challenge ./challenge
COPY data ./data

EXPOSE 8080

CMD ["sh", "-c", "uvicorn challenge.api:app --host 0.0.0.0 --port ${PORT:-8080}"]