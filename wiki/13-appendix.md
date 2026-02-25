## 13. Appendix

### 13.1 Sample API Request (cURL)

```bash
curl -X POST "https://your-app.railway.app/api/v1/recognize" \
  -H "Content-Type: multipart/form-data" \
  -F "image=@card_photo.jpg" \
  -F "model=clip" \
  -F "top_k=5"
```

### 13.2 Sample Environment Variables (.env.example)

```bash
# Application
ENV=development
LOG_LEVEL=debug
API_BASE_URL=http://localhost:8000

# Models
CLIP_MODEL_NAME=openai/clip-vit-base-patch32
CNN_MODEL_PATH=models/efficientnet_weights.pth
FAISS_INDEX_PATH=data/faiss_index.bin

# Performance
MAX_UPLOAD_SIZE_MB=10
INFERENCE_TIMEOUT_SEC=30
BATCH_SIZE=1

# Railway (auto-provided in production)
# PORT=8000
# RAILWAY_PUBLIC_DOMAIN=your-app.railway.app
```

### 13.3 Deployment Checklist

**Pre-Deployment:**
- [ ] All dependencies in pyproject.toml (Poetry) with pinned versions
- [ ] Environment variables documented in .env.example
- [ ] railway.toml configured (if using custom commands)
- [ ] Model weights and datasets in data/ directory or downloadable
- [ ] Health check endpoint implemented at /api/v1/health
- [ ] Unit tests passing (>80% coverage)
- [ ] README.md with setup instructions

**Railway Setup:**
- [ ] GitHub repository connected to Railway
- [ ] Service created and linked to repository
- [ ] Environment variables configured in Railway dashboard
- [ ] Volume created and mounted at /app/data (if needed)
- [ ] Public domain generated for service
- [ ] Custom domain configured (optional)
- [ ] Auto-deploy enabled for main branch

**Post-Deployment:**
- [ ] Health check endpoint returns 200 OK
- [ ] API endpoints respond correctly to test requests
- [ ] Streamlit UI loads and accepts image uploads
- [ ] Recognition results match expected accuracy
- [ ] Logs show no errors in Railway dashboard
- [ ] Response times meet performance targets (<500ms p50)
- [ ] SSL certificate verified (HTTPS works)

### 13.4 Current Project Structure (src layout)

```text
mtg-invasion-recognizer/
├── src/
│   ├── api/
│   │   └── main.py
│   ├── dataset/
│   │   └── download_cards.py
│   ├── recognizer/
│   │   └── build_index.py
│   └── ui/
│       └── app.py
├── data/
├── pyproject.toml
├── Makefile
└── README.md
```

### 13.5 Local Run Guide

#### Prerequisites

- Python 3.12
- Poetry installed

Install Poetry (if needed):

```bash
curl -sSL https://install.python-poetry.org | python3 -
```

#### Setup

From the repository root:

```bash
make install
```

#### Data and Index Preparation

```bash
make download-cards
make build-index
```

#### Run the Services Locally

Terminal 1 (FastAPI API):

```bash
make run-api
```

Terminal 2 (Streamlit):

```bash
make run-ui
```

#### Local URLs

- API health check: `http://localhost:8000/api/v1/health`
- FastAPI docs: `http://localhost:8000/docs`
- Streamlit UI: `http://localhost:8501`

#### Equivalent Poetry Commands

```bash
poetry install
poetry run python -m dataset.downloader
poetry run python -m recognizer.build_index
poetry run uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
poetry run streamlit run src/ui/app.py --server.port 8501
```

***
