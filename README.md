# 🏗️ Structural-Defects-Network-MLOps

An end-to-end MLOps pipeline for detecting concrete structural cracks using deep learning (ResNet and MobileNet) with PyTorch, Airflow for orchestration, and MLflow for experiment tracking.
In addition, we support

---

## 🔧 Prerequisites

- Python >= 3.10
- Docker + Docker Compose
- Optional: Virtualenv or conda

---

## 📦 Setup Instructions

### 1. Create a Virtual Environment (Recommended)

**Windows**

```bash
python -m venv env
.\env\Scripts\activate
```

**macOS/Linux**

```bash
python -m venv env
source env/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🗂️ Project Structure

```bash
python src/preprocess_set_seed.py
```

### 4. Data Artifacts

After running the preprocessing script:
The train, test, and validation datasets will be saved inside the artifact_folder/ directory.
The artifact_folder/ is excluded from version control and will not be uploaded to Git (as specified in .gitignore).

```kotlin
Structural-Defects-Network-MLOps/
├── Dataset/                  # Raw Data (Decks, Walls, Pavements)
├── airflow/
│   ├── dags/                 # Airflow DAGs and Python callables
│   │   ├── main_dag.py
│   │   └── callables/        # All python callables function in main dag
│   ├── config/               # airflow.cfg and other configs
│   ├── plugins/              # Airflow plugins (if needed)
│   ├── logs/                 # Airflow logs
│   ├── Dockerfile            # Custom Image for this project
│   └── docker-compose.yaml   # Docker Compose for Airflow stack
├── artifact_folder/          # Preprocessed datasets (train/val/test) – auto-generated after preprocessing
├── src/                      # CLI scripts for local training or upload
│   ├── scripts/              # All extra python scripts
│   ├── preprocess_set_seed.py # preprocessing scripts to run before using jupyter notebooks
│   ├── MobileNetV3.ipynb     # MobileNetv3 training steps
│   ├── resnet.ipynb          # ResNet18 training steps
│   └── docker-compose.yaml   # Docker Compose for Airflow stack
├── model/                    # Saved PyTorch model files                  # Airflow and pipeline logs
├── requirements.txt
└── README.md
```

---

## 🧹 Data Preprocessing

Run this step before training:

```bash
python src/preprocess_set_seed.py
```

This will generate `train/`, `val/`, and `test/` image folders + CSVs under `artifact_folder/`.

---

## 🐳 Run with Docker + Airflow

### 1. Add Environment Variables

Create a `.env` in the airflow folder root `airflow/`:

```env
AIRFLOW_UID=50000
CLOUDINARY_CLOUD_NAME=your_cloud_name
CLOUDINARY_API_KEY=your_key
CLOUDINARY_API_SECRET=your_secret
DATASET_PATH=<path>/<to>/your/projectStructural-Defects-Network-MLOps\Dataset
```

| _Reminder: this dataset path should be absolute path._

### 2. Start Airflow

```bash
docker compose up -d --build
```

Airflow UI → [http://localhost:8080](http://localhost:8080)
**Username / Password**: `airflow` / `airflow`

### 3. Stop Airflow

```bash
docker compose down --volumes --rmi all
```

---

## 📊 MLflow Tracking (Optional)

### Start MLflow UI locally:

```bash
mlflow ui --port 5000
```

MLflow UI → [http://localhost:5000](http://localhost:5000)

### Logs from training tasks in Airflow will be registered automatically (ResNet, MobileNet).

---

## ⚙️ Triggering the DAG

You can trigger the DAG manually from the Airflow UI, or let it run based on the monthly schedule (`@weekly`).

It checks:

- If new images have been uploaded to Cloudinary (e.g. under `Users/Cracked/`, `Users/Non-cracked/`)
- If >=10 new images are found → preprocess → train → log to MLflow
- Otherwise → skip training

---

## ✨ Notes

- The dataset must follow this structure:

  ```
  Dataset/
  ├── Decks/
  ├── Pavements/
  └── Walls/
  ```

- Preprocessing includes augmentation, resizing, and denoising filters.

- Models are saved to `opt/airflow/model/` and registered via MLflow.

- Metrics are saved using XCom and optionally pushed to Airflow Variables.

---

## 🖼️ Frontend Prediction Interface

This project includes a simple React-based **frontend web app** that enables:

- 🔍 Uploading images for **real-time prediction** (Cracked / Non-cracked)
- 🧠 **Model inference** using the latest trained model (ResNet or MobileNet)
- 📤 Submitting images with labels for **continuous learning**

---

### 📁 Frontend Directory Structure

```bash
app/
├── templates/
├── demo_run.ipynb                      # Contains REACT_APP_API_URL
├── Dockerfile
├── main.py
└── requirements.txt
```

---

### 🚀 Quick Start

1. Navigate to the frontend directory:

```bash
cd app
```

2. Build a Docker Image

```bash
docker build --tag 'Structure_prediction_UI'
```

3. Access the website:

```bash
localhost:8000
```

---

### 🌐 Features

- **Drag-and-Drop Upload** or file selector
- Calls backend API (`/predict`) to return:

  - Predicted label
  - Confidence score

- Option to submit the image to Cloudinary under:

  - `Users/Cracked/`
  - `Users/Non-cracked/`

- Those images will later be retrained during the next Airflow pipeline run

---

### 🧠 How It Works

- The frontend sends an image to a FastAPI
- The backend:

  - Loads the **latest model checkpoint**
  - Applies preprocessing
  - Performs a forward pass
  - Returns prediction + confidence

- Optionally, the backend can upload the labeled image to Cloudinary for future training

---

### 🧪 API Routes (Backend)

| Method | Endpoint | Description                        |
| ------ | -------- | ---------------------------------- |
| GET    | `/`      | Predicts the class of a user image |
| POST   | `/`      | Predict the class of a user image  |

---

### 📝 TODOs (Optional)

- Add a history view of past predictions
- Display class probabilities in a bar chart
- Show model version info

---

## 📄 License

[MIT License](LICENSE)
