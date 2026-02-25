## 1. Project Overview

### 1.1 Objective

Build a production-ready web application that identifies Magic: The Gathering cards from the Invasion block (Invasion, Planeshift, Apocalypse) using computer vision and deep learning techniques. The system accepts uploaded card images and returns accurate card identification with confidence scoring.

### 1.2 Scope

**In Scope:**
- Card recognition for 350 unique Invasion block cards
- Support for standard card orientations (portrait)
- Web-based interface with image upload
- REST API for programmatic access
- Confidence scoring and top-N predictions
- Card metadata retrieval (name, set, collector number, rarity)
- Three recognition approaches: CLIP embeddings + FAISS, CNN classification, perceptual hashing

**Out of Scope (Phase 1):**
- Real-time video stream recognition
- Recognition of non-English cards
- Handling of damaged, altered, or proxy cards
- Support for other MTG sets beyond Invasion block
- Mobile-native applications (iOS/Android)
- Bulk processing of multiple images
- Authentication and user account management

### 1.3 Target Users

- MTG collectors organizing Invasion block collections
- Card shop employees for inventory identification
- Players verifying card authenticity for tournaments
- Hobbyists building digital card databases

***
