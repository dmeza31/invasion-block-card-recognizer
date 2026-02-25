## 2. System Architecture

### 2.1 High-Level Architecture

```
┌─────────────────┐
│  Web Interface  │
│   (Streamlit)   │
└────────┬────────┘
         │
         │ HTTP/WebSocket
         │
┌────────▼────────┐
│   FastAPI       │
│   Backend       │
│  (REST API)     │
└────────┬────────┘
         │
    ┌────┴────┐
    │         │
┌───▼───┐ ┌──▼──────┐
│ CLIP  │ │   CNN   │
│ +FAISS│ │Classify │
└───┬───┘ └──┬──────┘
    │        │
    └────┬───┘
         │
┌────────▼────────┐
│  Card Dataset   │
│  (Images +      │
│   Metadata)     │
└─────────────────┘
```

### 2.2 Component Breakdown

#### 2.2.1 Card Dataset Module

**Purpose:** Manages the reference card image database and metadata

**Components:**
- `data/invasion_cards/` - Directory containing 350 reference card images (one per unique card)
- `data/metadata.json` - Card information database (name, set, collector number, rarity, type)
- `data/card_embeddings.npy` - Pre-computed CLIP embeddings for FAISS indexing
- `data/faiss_index.bin` - FAISS index file for similarity search

**Implementation Notes:**
- Images stored at consistent resolution (488×680 or scaled to 224×224 for models)
- Metadata schema includes: `card_id`, `name`, `set`, `collector_number`, `rarity`, `type_line`, `image_path`
- Dataset size: ~350 images × 200KB = ~70MB total
- FAISS index using HNSW algorithm for fast approximate nearest neighbor search

#### 2.2.2 CLIP Embedder + FAISS Index

**Purpose:** Embedding-based recognition using OpenAI's CLIP model

**Components:**
- `models/clip_embedder.py` - Wrapper for CLIP model inference
- `models/faiss_index.py` - FAISS index manager for similarity search
- `scripts/build_index.py` - Script to pre-compute embeddings and build FAISS index

**Implementation Notes:**
- CLIP model: `openai/clip-vit-base-patch32` (149M parameters)
- Image preprocessing: Resize to 224×224, normalize with CLIP's mean/std
- Embedding dimension: 512 (CLIP output)
- FAISS index type: `IndexHNSWFlat` with `M=16` links per layer
- Cosine similarity for distance metric
- Top-K retrieval: K=5 candidates with similarity scores

**Performance Targets:**
- Embedding generation: <100ms per image on CPU
- FAISS search: <10ms for top-5 retrieval
- Memory footprint: ~200MB (model) + ~1MB (FAISS index)

#### 2.2.3 CNN Classifier (EfficientNet)

**Purpose:** Direct classification approach using transfer learning

**Components:**
- `models/cnn_classifier.py` - EfficientNet-based classifier
- `models/efficientnet_weights.pth` - Fine-tuned model weights
- `training/train_classifier.py` - Training script with data augmentation

**Implementation Notes:**
- Base model: EfficientNet-B0 (5.3M parameters)
- Transfer learning: Replace final classification layer (1000 → 350 classes)
- Training approach: Freeze early layers, fine-tune final layers + new classifier head
- Data augmentation: Random rotation (±15°), brightness/contrast adjustment, horizontal flip
- Loss function: Cross-entropy with label smoothing (ε=0.1)
- Optimizer: AdamW with learning rate 1e-4, weight decay 1e-5
- Training dataset: 350 cards × 10 augmented versions = 3,500 training samples

**Performance Targets:**
- Inference time: <150ms per image on CPU
- Top-1 accuracy: >85% on clean test images
- Top-5 accuracy: >95% on clean test images
- Memory footprint: ~20MB (model weights)

#### 2.2.4 Perceptual Hashing (Optional Baseline)

**Purpose:** Fast approximate matching using image hashing

**Components:**
- `models/perceptual_hash.py` - pHash implementation using DCT
- Hash database stored in metadata.json

**Implementation Notes:**
- Algorithm: Discrete Cosine Transform (DCT) based perceptual hash
- Hash size: 64-bit binary hash
- Distance metric: Hamming distance (count differing bits)
- Threshold: <10 bits difference for match
- Use case: Fast filtering before running neural network models

**Performance Targets:**
- Hash computation: <10ms per image
- Hash comparison: <1ms per database lookup
- Best for: Near-duplicate detection, not robust to rotation/perspective

#### 2.2.5 FastAPI Backend

**Purpose:** REST API service for card recognition

**Components:**
- `api/main.py` - FastAPI application entry point
- `api/routes/recognize.py` - Recognition endpoint handlers
- `api/models/schemas.py` - Pydantic models for request/response validation
- `api/middleware/` - CORS, logging, error handling middleware

**Implementation Notes:**
- Framework: FastAPI with Uvicorn ASGI server
- Image upload: Multipart form-data, max size 10MB
- Response format: JSON with card predictions, confidence scores, metadata
- Error handling: Structured error responses with HTTP status codes
- Async processing: Background tasks for model inference to avoid blocking
- Health check endpoint: `/health` for monitoring

**Endpoints:**
- `POST /api/v1/recognize` - Upload image, get predictions
- `GET /api/v1/cards/{card_id}` - Retrieve card metadata by ID
- `GET /api/v1/health` - Service health check
- `GET /api/v1/docs` - OpenAPI/Swagger documentation

#### 2.2.6 Streamlit Web Interface

**Purpose:** User-facing web application

**Components:**
- `app.py` - Streamlit application
- `components/upload.py` - Image upload widget
- `components/results.py` - Results display with confidence bars
- `utils/api_client.py` - HTTP client for backend API calls

**Implementation Notes:**
- Upload widget: Supports JPEG, PNG, WebP formats
- Preview pane: Display uploaded image at 400px width
- Results display: Top-5 predictions with card images, names, confidence percentages
- Sidebar: Model selection (CLIP, CNN, pHash), confidence threshold slider
- Error handling: User-friendly error messages for upload failures, API errors

**Features:**
- Drag-and-drop image upload
- Real-time recognition results
- Interactive confidence threshold adjustment
- Card detail view with full metadata

***
