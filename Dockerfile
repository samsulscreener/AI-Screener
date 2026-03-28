FROM python:3.11-slim

WORKDIR /app

# System deps for lxml, newspaper3k, torch
RUN apt-get update && apt-get install -y \
    build-essential libxml2-dev libxslt1-dev \
    libjpeg-dev zlib1g-dev libpng-dev \
    curl git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p data/results logs

EXPOSE 8501

# Default: run dashboard
CMD ["streamlit", "run", "dashboard/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
