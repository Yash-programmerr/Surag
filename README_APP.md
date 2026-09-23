# SURAG Satellite Tile Search

This app provides text and image-similarity search over the existing `india_tiles` data,
metadata filters, a Leaflet map, and optional 2021-to-2025 change previews.

## Backend installation and development run

From the project root:

```bash
python3 -m pip install -r requirements_app.txt
uvicorn app:app --host 127.0.0.1 --port 8000
```

The backend uses `india_tiles` relative to the directory where it is started. It also starts when
the embeddings, change directory, thumbnails, or frontend build are absent.

Text search uses the existing `semantic_search.py` CLIP implementation. Install its optional ML
dependencies in the same Python environment before using text search:

```bash
python3 -m pip install torch transformers
```

The first text search may download the configured CLIP model; image-similarity search does not
load the text model.

## Frontend development run

In a second terminal:

```bash
cd frontend
npm install
npm run dev
```

The Vite server runs on `http://localhost:5173` and proxies `/api` and `/files` to the backend.

## Production build served by FastAPI

```bash
cd frontend
npm install
npm run build
cd ..
uvicorn app:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000`.

## API endpoints

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/api/health` | Data and embedding counts |
| GET | `/api/stats` | Category and year statistics |
| POST | `/api/search` | Text or tile-similarity search with filters |
| GET | `/api/tiles/{tile_id}` | Tile metadata and change summary |
| GET | `/api/tiles/{tile_id}/change` | Change summary, preview, and polygons |
| POST | `/api/reload` | Reload all data files and return health |

## Assumptions

- `india_tiles` remains next to the existing project modules and is the default data directory.
- The 2025 row with a valid thumbnail is the main row; only embedded tiles with an existing
  thumbnail participate in search.
- Change results are limited to status `ok` rows for 2021 and 2025.
- Invalid numeric fields are treated as missing and serialized as `null`, never as JSON `NaN`.
- The default text embedder imports `semantic_search` only when a text search is requested.
- Map imagery requires network access to the Esri or OpenStreetMap tile servers.
- Terminal checks cannot verify browser-only interactions such as clicking map rectangles, opening
  popups, or changing Leaflet base layers.
