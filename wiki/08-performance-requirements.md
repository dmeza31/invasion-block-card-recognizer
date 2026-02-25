## 8. Performance Requirements

### 8.1 Functional Requirements

| Requirement | Target | Priority |
|-------------|--------|----------|
| Card Recognition Accuracy (Top-1) | >85% | P0 |
| Card Recognition Accuracy (Top-5) | >95% | P0 |
| Supported Image Formats | JPEG, PNG, WebP | P0 |
| Maximum Image Size | 10MB | P0 |
| Concurrent Users | 10+ simultaneous uploads | P1 |
| API Uptime | >99.5% | P1 |

### 8.2 Non-Functional Requirements

| Requirement | Target | Priority |
|-------------|--------|----------|
| API Response Time (p50) | <500ms | P0 |
| API Response Time (p95) | <1000ms | P0 |
| Frontend Load Time | <2s | P1 |
| CLIP Embedding Generation | <100ms | P0 |
| FAISS Search Latency | <10ms | P0 |
| CNN Inference Time | <150ms | P0 |
| Model Memory Footprint | <300MB total | P1 |

### 8.3 Scalability Requirements

| Requirement | Target | Priority |
|-------------|--------|----------|
| Requests per Minute | 100 RPM | P1 |
| Peak Load Handling | 2x normal load | P2 |
| Database Query Time | <50ms | P1 |
| Storage Scalability | Support 10,000+ cards (future) | P2 |

***
