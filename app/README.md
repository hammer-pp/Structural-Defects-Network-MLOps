# How to launch the UI?

## 🐳 By Docker

1. Navigate to the frontend directory:

```bash
cd app
```

2. Build the docker image

```bash
docker build . --tag 'structure_prediction_ui'
```

3. Run the docker image and access the web

```bash
docker run -p 8000:8000 structure_prediction_ui
```

or use `docker compose up --build`

## 💨 By fastapi cli

1. Setup the virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

2. Install the dependencies

```bash
pip install -r requirements.txt
```

3. Access the website:

```bash
mkdir static
fastapi dev main.py
localhost:8000
```

---
