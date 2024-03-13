FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY src /app/src
COPY models /app/models

EXPOSE 8080

CMD ["python", "-m", "src.main"]