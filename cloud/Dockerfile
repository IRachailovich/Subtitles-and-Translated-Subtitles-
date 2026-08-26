FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app
COPY . /app
RUN pip install --no-cache-dir ".[cloud]"

EXPOSE 8000
CMD ["uvicorn", "subgen_cloud.api:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
