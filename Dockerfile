FROM python:3.10-slim

RUN apt-get update && apt-get install -y \
    gcc \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY dengue_pipeline/dashboard/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# IMPORTANT: Set this to the folder containing app.py
WORKDIR /app/dengue_pipeline/dashboard

EXPOSE 8501

# Shell form (no brackets) prevents the "not found" error
CMD streamlit run app.py --server.port=8501 --server.address=0.0.0.0 --server.headless=true --browser.gatherUsageStats=false