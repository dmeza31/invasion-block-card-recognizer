## 6. Technology Stack

### 6.1 Backend

| Component | Technology | Version |
|-----------|-----------|---------|
| Framework | FastAPI | 0.115+ |
| Server | Uvicorn | 0.32+ |
| ML Framework | PyTorch | 2.5+ |
| CLIP Model | transformers (Hugging Face) | 4.46+ |
| CNN Model | torchvision (EfficientNet) | 0.20+ |
| Vector Search | FAISS | 1.9+ |
| Image Processing | Pillow | 11.0+ |
| Data Validation | Pydantic | 2.9+ |
| HTTP Client | httpx | 0.28+ |

### 6.2 Frontend

| Component | Technology | Version |
|-----------|-----------|---------|
| Framework | Streamlit | 1.40+ |
| HTTP Client | requests | 2.32+ |

### 6.3 Development & Deployment

| Component | Technology | Version |
|-----------|-----------|---------|
| Language | Python | 3.11+ |
| Package Manager | pip / Poetry | - |
| Deployment Platform | Railway | - |
| Version Control | Git | - |
| CI/CD | Railway Auto-Deploy | - |

### 6.4 Dependencies File

**requirements.txt:**
```
fastapi==0.115.0
uvicorn[standard]==0.32.0
torch==2.5.0
torchvision==0.20.0
transformers==4.46.0
faiss-cpu==1.9.0
pillow==11.0.0
pydantic==2.9.0
pydantic-settings==2.6.0
python-multipart==0.0.12
httpx==0.28.0
streamlit==1.40.0
requests==2.32.0
numpy==1.26.4
```

***
