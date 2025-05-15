FROM apache/airflow:3.0.1

# Upgrade pip and setuptools first to avoid issues with Python 3.12
RUN python -m ensurepip --upgrade \
    && python -m pip install --upgrade pip setuptools

# Copy your requirements.txt
COPY requirements.txt /

# Install Airflow version and your requirements
RUN pip install --no-cache-dir "apache-airflow==${AIRFLOW_VERSION}" \
    -r /requirements.txt

# Explicitly install numpy 1.26.4 for Python 3.12 compatibility
RUN pip install --no-cache-dir numpy==1.26.4
