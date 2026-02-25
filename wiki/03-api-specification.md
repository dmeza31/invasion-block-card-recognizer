## 3. API Specification

### 3.1 Recognition Endpoint

**POST /api/v1/recognize**

Upload a card image and receive recognition predictions.

**Request:**
```
POST /api/v1/recognize
Content-Type: multipart/form-data

Parameters:
- image: file (required) - Card image file (JPEG, PNG, WebP)
- model: string (optional) - Recognition model: "clip", "cnn", "phash" (default: "clip")
- top_k: integer (optional) - Number of predictions to return (default: 5, max: 10)
```

**Response (Success - 200 OK):**
```json
{
  "predictions": [
    {
      "card_id": "inv_001",
      "name": "Absorb",
      "set": "Invasion",
      "collector_number": "226",
      "rarity": "rare",
      "type_line": "Instant",
      "confidence": 0.952,
      "image_url": "/static/cards/inv_001.jpg"
    },
    {
      "card_id": "inv_045",
      "name": "Counterspell",
      "set": "Invasion",
      "collector_number": "55",
      "rarity": "common",
      "type_line": "Instant",
      "confidence": 0.023,
      "image_url": "/static/cards/inv_045.jpg"
    }
  ],
  "model_used": "clip",
  "processing_time_ms": 127
}
```

**Response (Error - 400 Bad Request):**
```json
{
  "error": "Invalid image format",
  "detail": "Supported formats: JPEG, PNG, WebP"
}
```

**Response (Error - 500 Internal Server Error):**
```json
{
  "error": "Model inference failed",
  "detail": "CLIP model encountered an error during embedding generation"
}
```

### 3.2 Card Metadata Endpoint

**GET /api/v1/cards/{card_id}**

Retrieve full metadata for a specific card.

**Request:**
```
GET /api/v1/cards/inv_001
```

**Response (Success - 200 OK):**
```json
{
  "card_id": "inv_001",
  "name": "Absorb",
  "set": "Invasion",
  "collector_number": "226",
  "rarity": "rare",
  "type_line": "Instant",
  "mana_cost": "{W}{U}{U}",
  "oracle_text": "Counter target spell. You gain 3 life.",
  "artist": "Terese Nielsen",
  "image_url": "/static/cards/inv_001.jpg"
}
```

**Response (Error - 404 Not Found):**
```json
{
  "error": "Card not found",
  "detail": "No card with ID 'inv_999' exists in database"
}
```

### 3.3 Health Check Endpoint

**GET /api/v1/health**

Check service availability and model readiness.

**Response (Success - 200 OK):**
```json
{
  "status": "healthy",
  "models": {
    "clip": "loaded",
    "cnn": "loaded",
    "faiss": "loaded"
  },
  "timestamp": "2026-02-24T19:40:00Z"
}
```

### 3.4 OpenAPI Documentation

Interactive API documentation automatically generated at:
- Swagger UI: `https://<your-app>.railway.app/docs`
- ReDoc: `https://<your-app>.railway.app/redoc`
- OpenAPI JSON: `https://<your-app>.railway.app/openapi.json`

***
