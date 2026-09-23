from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path
from typing import Any, Callable

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from pyproj import Transformer

FLOAT_FIELDS = ("crop_frac", "built_frac", "open_frac", "water_frac", "valid_fraction",
                "ndvi_mean", "ndbi_mean", "lon", "lat")
INT_FIELDS = ("epsg", "year")
TEXT_TEMPLATE = "a satellite image of {}"


def clean(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(item) for item in value]
    return value


def number(value: Any, integer: bool = False) -> int | float | None:
    try:
        parsed = int(value) if integer else float(value)
        return parsed
    except (TypeError, ValueError):
        return None


class DataStore:
    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        self._transformers: dict[int, Transformer] = {}
        self.load()

    def load(self) -> None:
        self.ids: list[str] = []
        self.emb = np.zeros((0, 512), dtype=np.float32)
        self.main: dict[str, dict[str, Any]] = {}
        self.years: dict[str, list[int]] = {}
        self.change: dict[str, dict[str, Any]] = {}
        self.bounds: dict[str, list[list[float | None]]] = {}
        index = self.data_dir / "tiles_index.csv"
        rows: dict[str, list[dict[str, Any]]] = {}
        if index.exists():
            with index.open(newline="", encoding="utf-8") as stream:
                for raw in csv.DictReader(stream):
                    tile_id = raw.get("tile_id")
                    if not tile_id:
                        continue
                    row = self._convert(raw)
                    rows.setdefault(tile_id, []).append(row)
                    if row.get("year") is not None:
                        self.years.setdefault(tile_id, []).append(row["year"])
                    if row.get("year") == 2025 and raw.get("thumb"):
                        thumb = Path(raw["thumb"])
                        thumb_exists = thumb.exists() if thumb.is_absolute() else (
                            (self.data_dir.parent / thumb).exists() or (self.data_dir / thumb).exists()
                        )
                        if thumb_exists:
                            self.main[tile_id] = row
        self.years = {key: sorted(set(value)) for key, value in self.years.items()}
        emb_path = self.data_dir / "embeddings" / "embeddings.npy"
        ids_path = self.data_dir / "embeddings" / "tile_ids.json"
        if emb_path.exists() and ids_path.exists():
            try:
                emb = np.asarray(np.load(emb_path), dtype=np.float32)
                ids = json.loads(ids_path.read_text(encoding="utf-8"))
                keep = [i for i, tile_id in enumerate(ids) if i < len(emb) and tile_id in self.main]
                self.ids = [ids[i] for i in keep]
                self.emb = emb[keep]
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                self.ids, self.emb = [], np.zeros((0, 512), dtype=np.float32)
        for tile_id in self.ids:
            self.bounds[tile_id] = self._bounds(self.main[tile_id])
        summary = self.data_dir / "change" / "change_summary.csv"
        if summary.exists():
            with summary.open(newline="", encoding="utf-8") as stream:
                for raw in csv.DictReader(stream):
                    if raw.get("status") == "ok" and raw.get("year_a") == "2021" and raw.get("year_b") == "2025":
                        self.change[raw["tile_id"]] = self._convert_change(raw)

    @staticmethod
    def _convert(raw: dict[str, Any]) -> dict[str, Any]:
        row = dict(raw)
        for key in FLOAT_FIELDS:
            row[key] = number(raw.get(key))
        for key in INT_FIELDS:
            row[key] = number(raw.get(key), integer=True)
        return row

    @staticmethod
    def _convert_change(raw: dict[str, Any]) -> dict[str, Any]:
        row = dict(raw)
        for key in ("year_a", "year_b"):
            row[key] = number(raw.get(key), integer=True)
        for key in ("valid_fraction", "changed_fraction", "new_built_ha", "vegetation_loss_ha",
                    "vegetation_gain_ha", "water_change_ha", "other_change_ha", "total_change_ha",
                    "mean_dndvi", "mean_dndbi"):
            row[key] = number(raw.get(key))
        row["suspicious"] = str(raw.get("suspicious", "")).lower() == "true"
        return row

    def _bounds(self, row: dict[str, Any]) -> list[list[float | None]]:
        epsg = row.get("epsg")
        coords = [(row.get("x0"), row.get("y0")), (row.get("x1"), row.get("y0")),
                  (row.get("x1"), row.get("y1")), (row.get("x0"), row.get("y1"))]
        if epsg is None or any(x is None or y is None for x, y in coords):
            return [[None, None], [None, None]]
        transformer = self._transformers.setdefault(epsg, Transformer.from_crs(epsg, 4326, always_xy=True))
        points = [transformer.transform(x, y) for x, y in coords]
        lons, lats = zip(*points)
        return [[min(lats), min(lons)], [max(lats), max(lons)]]

    def row(self, tile_id: str) -> dict[str, Any] | None:
        return self.main.get(tile_id)


class Filters(BaseModel):
    category: list[str] | None = None
    min_built: float | None = None
    max_built: float | None = None
    min_crop: float | None = None
    min_open: float | None = None
    max_water: float | None = None
    min_valid: float = 0.9


class SearchRequest(BaseModel):
    text: str | None = None
    like_tile_id: str | None = None
    k: int = Field(default=12)
    filters: Filters = Field(default_factory=Filters)


def apply_filters(store: DataStore, filters: Filters) -> np.ndarray:
    mask = np.ones(len(store.ids), dtype=bool)
    for i, tile_id in enumerate(store.ids):
        row = store.main[tile_id]
        if filters.category and row.get("category") not in filters.category:
            mask[i] = False
            continue
        checks = (("built_frac", filters.min_built, lambda a, b: a >= b),
                  ("built_frac", filters.max_built, lambda a, b: a <= b),
                  ("crop_frac", filters.min_crop, lambda a, b: a >= b),
                  ("open_frac", filters.min_open, lambda a, b: a >= b),
                  ("water_frac", filters.max_water, lambda a, b: a <= b),
                  ("valid_fraction", filters.min_valid, lambda a, b: a >= b))
        for key, bound, predicate in checks:
            value = row.get(key)
            if bound is not None and (value is None or not predicate(value, bound)):
                mask[i] = False
                break
    return mask


def create_app(data_dir: Path = Path("india_tiles"), text_embedder: Callable | None = None) -> FastAPI:
    store = DataStore(Path(data_dir))
    app = FastAPI(title="SURAG Tile Search")
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
    app.mount("/files/thumbs", StaticFiles(directory=str(store.data_dir / "thumbs"), check_dir=False), name="thumbs")
    app.mount("/files/change", StaticFiles(directory=str(store.data_dir / "change"), check_dir=False), name="change")
    dist = Path(__file__).parent / "frontend" / "dist"

    def health() -> dict[str, Any]:
        return {"status": "ok", "tiles_indexed": len(store.main), "tiles_embedded": len(store.ids),
                "tiles_with_change": len(store.change)}

    def public_result(tile_id: str, score: float, rank: int) -> dict[str, Any]:
        row = store.main[tile_id]
        return clean({"rank": rank, "tile_id": tile_id, "score": float(score),
                      "category": row.get("category"), "lat": row.get("lat"), "lon": row.get("lon"),
                      **{key: row.get(key) for key in ("crop_frac", "built_frac", "open_frac", "water_frac", "valid_fraction", "ndvi_mean")},
                      "thumb_url": f"/files/thumbs/{tile_id}.jpg", "bounds": store.bounds[tile_id],
                      "has_change": tile_id in store.change,
                      "change_total_ha": store.change.get(tile_id, {}).get("total_change_ha")})

    @app.get("/api/health")
    def api_health() -> dict[str, Any]:
        return health()

    @app.get("/api/stats")
    def api_stats() -> dict[str, Any]:
        categories: dict[str, int] = {}
        for row in store.main.values():
            categories[row["category"]] = categories.get(row["category"], 0) + 1
        years = {str(year): sum(year in values for values in store.years.values()) for year in (2021, 2025)}
        return {"embedded_by_category": categories, "with_change": len(store.change), "years": years}

    @app.post("/api/search")
    def api_search(request: SearchRequest) -> dict[str, Any]:
        has_text = request.text is not None
        has_like = request.like_tile_id is not None
        if has_text == has_like:
            raise HTTPException(400, "Provide exactly one of text or like_tile_id")
        if has_text and not request.text.strip():
            raise HTTPException(400, "Text query cannot be empty")
        if has_like and request.like_tile_id not in store.ids:
            raise HTTPException(404, "Tile is not embedded")
        if not len(store.ids):
            return {"query": {"text": request.text, "like_tile_id": request.like_tile_id, "k": max(1, min(request.k, 100))},
                    "searched_tiles": 0, "results": []}
        try:
            if has_text:
                embed = text_embedder
                if embed is None:
                    from semantic_search import embed_texts
                    embed = embed_texts
                query = np.asarray(embed([TEXT_TEMPLATE.format(request.text.strip())]))[0]
            else:
                query = store.emb[store.ids.index(request.like_tile_id)]
        except Exception as exc:
            if has_text:
                raise HTTPException(503, f"Text search is unavailable: {str(exc)[:120]}")
            raise
        scores = store.emb @ query
        mask = apply_filters(store, request.filters)
        if has_like:
            mask[store.ids.index(request.like_tile_id)] = False
        valid = np.where(mask)[0]
        k = min(max(1, min(request.k, 100)), len(valid))
        order = valid[np.argsort(scores[valid])[::-1][:k]] if k else []
        results = [public_result(store.ids[index], scores[index], rank) for rank, index in enumerate(order, 1)]
        return clean({"query": {"text": request.text, "like_tile_id": request.like_tile_id, "k": k},
                      "searched_tiles": len(store.ids), "results": results})

    @app.get("/api/tiles/{tile_id}")
    def api_tile(tile_id: str) -> dict[str, Any]:
        row = store.row(tile_id)
        if row is None:
            raise HTTPException(404, "Tile not found")
        return clean({"tile_id": tile_id, "category": row.get("category"), "lat": row.get("lat"), "lon": row.get("lon"),
                      "epsg": row.get("epsg"), "smod_code": row.get("smod_code"),
                      **{key: row.get(key) for key in ("crop_frac", "built_frac", "open_frac", "water_frac", "valid_fraction", "ndvi_mean", "ndbi_mean")},
                      "years_available": store.years.get(tile_id, []), "thumb_url": f"/files/thumbs/{tile_id}.jpg",
                      "bounds": store.bounds.get(tile_id), "maps_url": f"https://www.google.com/maps/@{row.get('lat')},{row.get('lon')},15z/data=!3m1!1e3",
                      "change": store.change.get(tile_id)})

    @app.get("/api/tiles/{tile_id}/change")
    def api_change(tile_id: str) -> dict[str, Any]:
        summary = store.change.get(tile_id)
        if summary is None:
            raise HTTPException(404, "No change result for this tile")
        category = summary.get("category", store.main.get(tile_id, {}).get("category", ""))
        base = store.data_dir / "change" / str(category)
        polygons_path = base / f"{tile_id}_2021_2025_polygons.geojson"
        polygons = {"type": "FeatureCollection", "features": []}
        if polygons_path.exists():
            polygons = json.loads(polygons_path.read_text(encoding="utf-8"))
        return clean({"summary": summary, "preview_url": f"/files/change/{category}/{tile_id}_2021_2025_preview.png",
                      "polygons": polygons})

    @app.post("/api/reload")
    def api_reload() -> dict[str, Any]:
        store.load()
        return health()

    @app.get("/")
    def root() -> Any:
        if dist.exists() and (dist / "index.html").exists():
            from fastapi.responses import FileResponse
            return FileResponse(dist / "index.html")
        return JSONResponse({"detail": "Frontend not built yet. Run: cd frontend && npm install && npm run build"})

    app.state.store = store
    if dist.exists():
        app.mount("/", StaticFiles(directory=str(dist), html=True), name="frontend")
    return app


if os.environ.get("SURAG_NO_AUTOAPP") != "1":
    app = create_app()
