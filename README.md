# Structural-Defects-Network-MLOps

## Prerequisites

Python version >= 3.10

## 📦 Setup Instructions

### 1. (Optional) Create a Virtual Environment

We recommend using a virtual environment to manage dependencies.

**For Windows:**

```bash
python -m venv env
.\env\Scripts\activate
```

**For macOS/Linux:**

```bash
python -m venv env
source env/bin/activate
```

### 2. Install Required Packages

Install all required Python packages using the requirements.txt file:

```bash
pip install -r requirements.txt
```

### 3. Run Preprocessing

Run the preprocessing script before training or evaluation:

```bash
python src/preprocess.py
```

### 4. Data Artifacts

After running the preprocessing script:
The train, test, and validation datasets will be saved inside the artifact_folder/ directory.
The artifact_folder/ is excluded from version control and will not be uploaded to Git (as specified in .gitignore).

```kotlin
Structural-Defects-Network-MLOps/
├── Dataset             # Raw Data
├── src/
│   └── preprocess.py/
├── requirements.txt
├── .gitignore
├── README.md
└── artifact_folder/     # Contains train/test/val data (ignored by Git)
```

## Start a docker for Airflow

### 5. (Optional) Set Environment Variables

Create a `.env` file in the project root to store environment variables (such as database URLs, API keys, etc.).  
Example `.env` file:

```env
AIRFLOW_UID=501
CLOUDINARY_CLOUD_NAME=your_cloud_name
CLOUDINARY_API_KEY=your_api_key
CLOUDINARY_API_SECRET=your_api_secret
DATASET_PATH=path/to/your/dataset
```

### Start docker

docker compose up -d --build

username and password are airflow and airflow

### Stop docker

docker compose down --volumes --rmi all
