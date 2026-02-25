## 4. User Interface

### 4.1 Streamlit Application Layout

**Page Structure:**
```
┌──────────────────────────────────────┐
│        MTG Card Recognizer           │
│         (Header)                     │
├──────────────────────────────────────┤
│ ┌────────┐ ┌────────────────────┐  │
│ │Sidebar │ │   Main Panel       │  │
│ │        │ │                    │  │
│ │Model   │ │  ┌──────────────┐ │  │
│ │Select  │ │  │ Upload Zone  │ │  │
│ │        │ │  │ (Drag/Drop)  │ │  │
│ │Conf.   │ │  └──────────────┘ │  │
│ │Thresh. │ │                    │  │
│ │        │ │  ┌──────────────┐ │  │
│ │Top-K   │ │  │Image Preview │ │  │
│ │        │ │  └──────────────┘ │  │
│ │        │ │                    │  │
│ │        │ │  ┌──────────────┐ │  │
│ │        │ │  │  Results     │ │  │
│ │        │ │  │  (Top-5)     │ │  │
│ │        │ │  └──────────────┘ │  │
│ └────────┘ └────────────────────┘  │
└──────────────────────────────────────┘
```

**Sidebar Controls:**
- Model selection radio buttons (CLIP, CNN, pHash)
- Confidence threshold slider (0.0 - 1.0, default 0.5)
- Top-K slider (1-10, default 5)
- "Clear Results" button

**Main Panel Sections:**
1. **Upload Zone** - Streamlit file_uploader with drag-and-drop support
2. **Image Preview** - Display uploaded image at 400px width
3. **Recognition Button** - "Recognize Card" button triggers API call
4. **Results Display** - Card grid showing:
   - Card image thumbnail
   - Card name and set
   - Confidence bar (color-coded: green >0.8, yellow 0.5-0.8, red <0.5)
   - "View Details" expander with full metadata

### 4.2 User Flow

1. User lands on homepage with upload zone
2. User drags/drops or selects card image file
3. Image preview appears below upload zone
4. User optionally adjusts sidebar settings (model, threshold, top-K)
5. User clicks "Recognize Card" button
6. Loading spinner appears during API call
7. Results display with top-K predictions sorted by confidence
8. User can click card to expand full metadata
9. User can upload another image (previous results remain visible for comparison)

### 4.3 Error Handling

**Client-Side Validation:**
- File size limit: 10MB (error message if exceeded)
- File format check: JPEG, PNG, WebP only
- Image dimension check: Min 100×100, max 4000×4000

**API Error Display:**
- Connection errors: "Unable to reach recognition service. Please try again."
- Timeout errors: "Request timed out. Please try with a smaller image."
- Server errors: "Recognition service encountered an error. Please try again later."

***
