# FraudShield API image. Build: docker build -t fraudshield .   Run: docker run -p 8000:8000 fraudshield
# Match this to the Python version you trained with (check: python --version)
FROM python:3.12-slim

# XGBoost needs the OpenMP runtime, which the slim image does not include
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Code, sample policy docs and the trained model (models/ must already be trained)
COPY src ./src
COPY docs ./docs
COPY models ./models

EXPOSE 8000
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]