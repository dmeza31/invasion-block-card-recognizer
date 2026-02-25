## 7. Deployment on Railway

### 7.1 Railway Platform Overview

Railway is a modern Platform-as-a-Service (PaaS) that simplifies deployment through automatic build detection, environment management, and instant scaling. Unlike Docker-based deployment, Railway uses Nixpacks, an intelligent buildpack system that automatically detects your application type and configures the build process.[^1][^2][^3][^4]

**Key Railway Features:**
- **Automatic Build Detection**: Detects Python apps via `requirements.txt`, `pyproject.toml`, or `main.py`[^5]
- **Zero-Config Deployment**: No Dockerfile required for standard Python applications[^1]
- **Git Integration**: Auto-deploys on GitHub push[^1]
- **Environment Variables**: Secure variable management with sealed secrets[^6]
- **Persistent Storage**: Volume mounting for model weights and datasets[^7]
- **Public Domains**: Automatic HTTPS with custom domain support[^8]

### 7.2 Deployment Configuration

#### 7.2.1 Project Structure for Railway

Ensure your repository root contains:
```
mtg-invasion-recognizer/
├── src/
│   ├── api/
│   │   ├── __init__.py
│   │   ├── main.py
│   │   └── routes/
│   ├── dataset/
│   │   ├── __init__.py
│   │   ├── download_cards.py
│   │   └── augment.py
│   ├── recognizer/
│   │   ├── __init__.py
│   │   ├── clip_embedder.py
│   │   ├── faiss_index.py
│   │   └── build_index.py
│   └── ui/
│       ├── __init__.py
│       └── app.py
├── data/
│   ├── invasion_cards/
│   ├── metadata.json
│   ├── card_embeddings.npy
│   └── faiss_index.bin
├── pyproject.toml
├── Makefile
├── railway.toml (optional)
├── .env.example
└── README.md
```

#### 7.2.2 Railway Configuration File (Optional)

While Railway auto-detects Python apps, you can customize the build/deploy process with `railway.toml`:[^9][^10]

**railway.toml:**
```toml
[build]
builder = "nixpacks"
buildCommand = "pip install poetry && poetry install --no-interaction --no-ansi"

[deploy]
startCommand = "poetry run uvicorn api.main:app --host 0.0.0.0 --port $PORT"
healthcheckPath = "/api/v1/health"
healthcheckTimeout = 300
restartPolicyType = "on_failure"
restartPolicyMaxRetries = 10
```

**Alternative for Streamlit:**
```toml
[deploy]
startCommand = "poetry run streamlit run src/ui/app.py --server.port $PORT --server.address 0.0.0.0"
```

**Note:** Railway automatically sets `$PORT` environment variable. Your application must listen on this port.[^6]

### 7.3 Environment Variables

Railway provides a secure variable management system with shared variables, sealed secrets, and service-specific configuration.[^6]

#### 7.3.1 Required Variables

Set these in Railway's Service Variables tab:

```bash
# Application Settings
ENV=production
LOG_LEVEL=info
API_BASE_URL=https://your-app.railway.app

# Model Configuration
CLIP_MODEL_NAME=openai/clip-vit-base-patch32
CNN_MODEL_PATH=/app/models/efficientnet_weights.pth
FAISS_INDEX_PATH=/app/data/faiss_index.bin

# Performance Tuning
MAX_UPLOAD_SIZE_MB=10
INFERENCE_TIMEOUT_SEC=30
BATCH_SIZE=1

# Railway-provided variables (automatic)
# RAILWAY_PUBLIC_DOMAIN - Your app's public URL
# RAILWAY_PRIVATE_DOMAIN - Internal service domain
# PORT - Port your app should listen on
```

#### 7.3.2 Setting Variables in Railway

**Method 1: Dashboard (Recommended)**
1. Navigate to your service → Variables tab
2. Click "New Variable"
3. Enter variable name and value
4. Click "Add" and deploy changes[^6]

**Method 2: Raw Editor (Bulk Import)**
1. Click "RAW Editor" in Variables tab
2. Paste `.env` file contents
3. Railway auto-detects variables from `.env.example` in your repo[^6]

**Method 3: Railway CLI**
```bash
railway variables set ENV=production
railway variables set LOG_LEVEL=info
```

#### 7.3.3 Sealed Variables for Secrets

For API keys or sensitive data, use sealed variables that are never visible in UI/API:[^6]

1. Add variable normally
2. Click 3-dot menu → "Seal"
3. Value is encrypted and hidden (cannot be unsealed)

### 7.4 Persistent Storage with Volumes

Railway volumes provide persistent storage that survives deployments.[^7]

#### 7.4.1 Creating a Volume

**Via Command Palette:**
1. Press `⌘K` (Mac) or `Ctrl+K` (Windows/Linux)
2. Type "New Volume" → Enter
3. Select your service
4. Set mount path: `/app/data`[^7]

**Via Right-Click Menu:**
1. Right-click project canvas
2. Select "New Volume"
3. Follow prompts

#### 7.4.2 Volume Configuration

**Mount Path:** `/app/data`
- Nixpacks places your app in `/app` directory[^7]
- Models and datasets stored at `/app/data/` persist across deployments
- Volume is mounted at runtime, not during build[^7]

**Important:** Volumes are not available during build time. Pre-download models/data to volume using a startup script or include in repository.

**Volume Variables (Auto-provided):**
```bash
RAILWAY_VOLUME_NAME=your-volume-name
RAILWAY_VOLUME_MOUNT_PATH=/app/data
```

#### 7.4.3 Populating the Volume

**Option 1: Include data in repository** (< 100MB recommended)
```
data/
├── invasion_cards/     # 350 card images
├── metadata.json
├── card_embeddings.npy
└── faiss_index.bin
```

**Option 2: Download on first startup** (for large datasets)
```python
# api/main.py startup event
@app.on_event("startup")
async def download_models():
    if not os.path.exists("/app/data/faiss_index.bin"):
        # Download from cloud storage
        download_from_s3("faiss_index.bin", "/app/data/")
```

**Option 3: Pre-deploy script** (railway.toml)
```toml
[deploy]
preDeployCommand = ["python scripts/setup_data.py"]
startCommand = "uvicorn api.main:app --host 0.0.0.0 --port $PORT"
```

### 7.5 Deployment Steps

#### 7.5.1 Initial Deployment from GitHub

1. **Push Code to GitHub**
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git remote add origin https://github.com/yourusername/mtg-card-recognition.git
   git push -u origin main
   ```

2. **Create Railway Project**
   - Go to [Railway Dashboard](https://railway.com)
   - Click "New Project"
   - Select "Deploy from GitHub repo"
   - Authorize Railway to access your GitHub account
   - Select your repository[^1]

3. **Railway Auto-Detection**
   - Railway scans repository and detects Python application
   - Automatically selects Nixpacks builder
   - Installs dependencies from `requirements.txt`
   - Detects FastAPI/Streamlit and configures start command[^4]

4. **Configure Environment Variables**
   - Go to Service → Variables tab
   - Add required variables (see Section 7.3.1)
   - Click "Deploy" to apply changes[^6]

5. **Create Volume (if needed)**
   - Press `⌘K` → "New Volume"
   - Mount path: `/app/data`
   - Attach to service[^7]

6. **Generate Public Domain**
   - Go to Service → Settings → Networking
   - Click "Generate Domain"
   - Your app is now accessible at `https://your-app.railway.app`[^8][^1]

#### 7.5.2 Deployment from Railway CLI

**Install Railway CLI:**
```bash
# macOS/Linux
curl -fsSL https://railway.app/install.sh | sh

# Windows
powershell -c "irm https://railway.app/install.ps1 | iex"
```

**Deploy Steps:**
```bash
# Authenticate
railway login

# Link to existing project (or create new)
railway link

# Deploy
railway up

# View logs
railway logs

# Open in browser
railway open
```

#### 7.5.3 Continuous Deployment

Railway automatically deploys on every push to your connected GitHub branch:[^1]

1. Make code changes locally
2. Commit and push to GitHub:
   ```bash
   git add .
   git commit -m "Update recognition model"
   git push origin main
   ```
3. Railway detects push, builds, and deploys automatically
4. Monitor deployment in Railway dashboard

**Disable auto-deploy (optional):**
- Service → Settings → Source → Disable "Auto-Deploy"

### 7.6 Custom Domain Setup

Add a custom domain for production use:[^11][^8]

1. **In Railway Dashboard:**
   - Service → Settings → Networking
   - Click "+ Custom Domain"
   - Enter your domain: `mtg-recognition.yourdomain.com`
   - Railway provides CNAME target: `g05ns7.up.railway.app`

2. **In Your DNS Provider (Cloudflare, Namecheap, etc.):**
   - Create CNAME record:
     ```
     Name: mtg-recognition
     Type: CNAME
     Target: g05ns7.up.railway.app
     TTL: Auto
     ```

3. **Wait for Verification:**
   - Railway verifies DNS propagation
   - Automatically issues SSL certificate
   - Green checkmark appears when ready

**For Root Domain (apex):**
- Use CNAME Flattening or ALIAS record (provider-dependent)
- Example: `yourdomain.com` → Railway CNAME target

### 7.7 Multi-Service Deployment (API + Frontend)

Deploy FastAPI backend and Streamlit frontend as separate services:

#### 7.7.1 Service 1: FastAPI Backend

**railway.toml:**
```toml
[build]
builder = "nixpacks"

[deploy]
startCommand = "uvicorn api.main:app --host 0.0.0.0 --port $PORT"
healthcheckPath = "/api/v1/health"
```

**Variables:**
```bash
SERVICE_TYPE=api
ENV=production
```

**Domain:** `api.your-app.railway.app`

#### 7.7.2 Service 2: Streamlit Frontend

**railway.toml:**
```toml
[deploy]
startCommand = "streamlit run app.py --server.port $PORT --server.address 0.0.0.0"
```

**Variables:**
```bash
SERVICE_TYPE=frontend
API_BASE_URL=${{ api.RAILWAY_PUBLIC_DOMAIN }}
```

**Domain:** `app.your-app.railway.app`

**Linking Services:**
- Use reference variables to access backend URL: `${{ api.RAILWAY_PUBLIC_DOMAIN }}`[^6]
- Or use Railway's private networking: `${{ api.RAILWAY_PRIVATE_DOMAIN }}`

### 7.8 Monitoring and Logging

**Railway Dashboard:**
- Service → Deployments tab: View build/deploy logs
- Service → Metrics tab: CPU, memory, network usage
- Service → Logs tab: Real-time application logs

**Application Logging:**
```python
import logging

# Configure logging for Railway
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
```

**Health Check Endpoint:**
Railway monitors `/api/v1/health` endpoint (configured in railway.toml). If endpoint returns non-2xx status, Railway restarts the service based on restart policy.

### 7.9 Scaling and Performance

**Automatic Scaling:**
- Railway automatically scales based on demand
- No manual configuration required for basic scaling

**Resource Limits (Starter Plan):**
- 0.5 GB RAM per service
- 1 vCPU per service
- 0.5 GB volume storage[^12]

**Optimization Tips:**
1. **Use Volume for Large Files**: Store models/datasets on volume instead of in Docker image
2. **Lazy Model Loading**: Load models on first request, not at startup
3. **Optimize Inference**: Use CPU-optimized models (ONNX, quantization) for faster inference
4. **Caching**: Cache FAISS embeddings in memory to avoid repeated disk reads

### 7.10 Cost Estimation

**Railway Pricing Model (Usage-Based):**[^12]
- **Starter Plan**: $5/month + usage
  - $5 included credits per month
  - Usage charges: Memory, CPU, storage, network egress
- **Pro Plan**: $20/month + usage
  - $20 included credits per month

**Estimated Monthly Cost for This Project:**
- **Compute**: ~$3-5/month (0.5GB RAM, light traffic)
- **Storage**: ~$0.50/month (0.1GB volume)
- **Network**: ~$1-2/month (egress for image downloads)
- **Total**: ~$5-8/month

**Free Trial:**
- Railway offers $5 credits for 30 days to try the platform[^12]
- No credit card required for trial

***
