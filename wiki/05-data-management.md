## 5. Data Management

### 5.1 Dataset Structure

```
data/
├── invasion_cards/
│   ├── inv_001_absorb.jpg
│   ├── inv_002_aether_rift.jpg
│   ├── ...
│   └── apc_143_worldly_counsel.jpg
├── metadata.json
├── card_embeddings.npy
└── faiss_index.bin
```

### 5.2 Metadata Schema

**metadata.json Structure:**
```json
{
  "cards": [
    {
      "card_id": "inv_001",
      "name": "Absorb",
      "set": "Invasion",
      "set_code": "INV",
      "collector_number": "226",
      "rarity": "rare",
      "type_line": "Instant",
      "mana_cost": "{W}{U}{U}",
      "cmc": 3,
      "oracle_text": "Counter target spell. You gain 3 life.",
      "colors": ["W", "U"],
      "color_identity": ["W", "U"],
      "artist": "Terese Nielsen",
      "image_path": "data/invasion_cards/inv_001_absorb.jpg",
      "perceptual_hash": "a1b2c3d4e5f67890"
    }
  ],
  "last_updated": "2026-02-24T19:00:00Z",
  "total_cards": 350
}
```

### 5.3 Storage Requirements

**Development Environment:**
- Card images: ~70MB (350 cards × 200KB avg)
- CLIP embeddings: ~700KB (350 cards × 512 dims × 4 bytes)
- FAISS index: ~1MB
- CNN model weights: ~20MB
- Total: ~92MB

**Production Environment:**
- Same as development
- Consider Railway Volume for persistence (see Section 7.3)

***
