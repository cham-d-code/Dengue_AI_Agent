# Use a slim Python image for smaller footprint
FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory inside the container
WORKDIR /app

# 1. Copy requirements from the subfolder and install
COPY dengue_pipeline/dashboard/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 2. Copy the entire project structure so app.py can find the data folder
COPY . .

# 3. Set the working directory to where app.py actually lives
# This ensures Streamlit finds app.py immediately
WORKDIR /app/dengue_pipeline/dashboard

# Expose the Streamlit port
EXPOSE 8501

# Run Streamlit with headless flags to prevent Exited(0)
CMD streamlit run app.py --server.port=8501 --server.address=0.0.0.0 --server.headless=true --browser.gatherUsageStats=false