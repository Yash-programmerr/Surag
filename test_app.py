import csv
import json
import os
from pathlib import Path

os.environ["SURAG_NO_AUTOAPP"] = "1"

import numpy as np
from PIL import Image
from fastapi.testclient import TestClient

from app import create_app


FIELDS = ["tile_id", "category", "year", "path", "thumb", "lon", "lat", "epsg", "x0", "y0", "x1", "y1",
          "smod_code", "crop_frac", "built_frac", "open_frac", "water_frac", "valid_fraction", "ndvi_mean",
          "ndbi_mean", "n_scenes", "mean_scene_cloud_pct", "bands", "downloaded_utc"]
CATEGORIES = ["agriculture", "city_structures", "open_land", "urban_areas"]


def build_data(root: Path) -> tuple[Path, dict[str, np.ndarray]]:
    (root / "thumbs").mkdir(parents=True)
    (root / "embeddings").mkdir()
    (root / "change" / "agriculture").mkdir(parents=True)
    centres = {category: np.random.default_rng(index).normal(size=512).astype(np.float32)
               for index, category in enumerate(CATEGORIES)}
    centres = {key: value / np.linalg.norm(value) for key, value in centres.items()}
    rows, ids, vectors = [], [], []
    for i in range(40):
        tile_id, category = f"32643_1_{i}", CATEGORIES[i % 4]
        x0, y0 = 400000 + i * 2560, 2600000
        fractions = {"crop_frac": .85 if category == "agriculture" else .05,
                     "built_frac": .6 if category == "city_structures" else (.25 if category == "urban_areas" else .05),
                     "open_frac": .8 if category == "open_land" else .05, "water_frac": .02}
        thumb = root / "thumbs" / f"{tile_id}.jpg"
        Image.new("RGB", (256, 256), (i, 100, 150)).save(thumb)
        row = [tile_id, category, 2025, "", str(thumb), 77.4 + i * .01, 23.3, 32643, x0, y0, x0 + 2560, y0 + 2560, 1,
               fractions["crop_frac"], fractions["built_frac"], fractions["open_frac"], fractions["water_frac"], .99, .2, .1, 1, 0, "RGB", ""]
        rows.append(row)
        if i < 10:
            rows.append([*row[:2], 2021, *row[3:]])
        ids.append(tile_id)
        vector = centres[category] + .3 * np.random.default_rng(i).normal(size=512)
        vectors.append((vector / np.linalg.norm(vector)).astype(np.float32))
    with (root / "tiles_index.csv").open("w", newline="") as stream:
        writer = csv.writer(stream); writer.writerow(FIELDS); writer.writerows(rows)
    np.save(root / "embeddings" / "embeddings.npy", np.stack(vectors))
    (root / "embeddings" / "tile_ids.json").write_text(json.dumps(ids))
    summary = ["32643_1_0", "agriculture", 2021, 2025, "ok", .99, .1, 1, 1, 1, 1, .1, 3.2, .1, .2, "False", "", "", ""]
    with (root / "change" / "change_summary.csv").open("w", newline="") as stream:
        writer = csv.writer(stream); writer.writerow(["tile_id", "category", "year_a", "year_b", "status", "valid_fraction", "changed_fraction", "new_built_ha", "vegetation_loss_ha", "vegetation_gain_ha", "water_change_ha", "other_change_ha", "total_change_ha", "mean_dndvi", "mean_dndbi", "suspicious", "mask_path", "preview_path", "polygons_path"]); writer.writerow(summary)
    preview = root / "change" / "agriculture" / "32643_1_0_2021_2025_preview.png"
    Image.new("RGB", (768, 256), "white").save(preview)
    polygon = {"type": "FeatureCollection", "features": [{"type": "Feature", "properties": {"tile_id": "32643_1_0", "category": "agriculture", "class_code": 1, "class_name": "new_built", "area_m2": 32000, "year_a": 2021, "year_b": 2025}, "geometry": {"type": "Polygon", "coordinates": [[[77.4, 23.3], [77.41, 23.3], [77.41, 23.31], [77.4, 23.3]]]}}]}
    (preview.parent / "32643_1_0_2021_2025_polygons.geojson").write_text(json.dumps(polygon))
    return root, centres


def client_factory(tmp_path):
    data, centres = build_data(tmp_path)
    def embed(texts):
        text = texts[0]
        category = "agriculture" if "farm" in text else "city_structures" if "build" in text else "open_land" if "barren" in text else "urban_areas" if "suburb" in text else "agriculture"
        return np.stack([centres[category]])
    return TestClient(create_app(data, embed))


def test_health(tmp_path): assert client_factory(tmp_path).get("/api/health").json()["tiles_embedded"] == 40
def test_stats(tmp_path):
    body = client_factory(tmp_path).get("/api/stats").json()
    assert body["embedded_by_category"] == {key: 10 for key in CATEGORIES} and body["years"]["2021"] == 10
def test_search_text_returns_matching_category(tmp_path):
    body = client_factory(tmp_path).post("/api/search", json={"text": "farmland", "k": 10}).json()
    assert len(body["results"]) == 10 and [r["rank"] for r in body["results"]] == list(range(1, 11))
    assert sum(r["category"] == "agriculture" for r in body["results"]) >= 8
def test_filters(tmp_path):
    client = client_factory(tmp_path)
    assert all(r["category"] == "city_structures" for r in client.post("/api/search", json={"text": "farmland", "filters": {"category": ["city_structures"]}}).json()["results"])
    assert all(r["built_frac"] >= .5 for r in client.post("/api/search", json={"text": "farmland", "filters": {"min_built": .5}}).json()["results"])
def test_search_like_excludes_self(tmp_path):
    assert "32643_1_5" not in [r["tile_id"] for r in client_factory(tmp_path).post("/api/search", json={"like_tile_id": "32643_1_5"}).json()["results"]]
def test_search_validation(tmp_path):
    client = client_factory(tmp_path)
    assert client.post("/api/search", json={}).status_code == 400 and client.post("/api/search", json={"text": "x", "like_tile_id": "x"}).status_code == 400
    assert client.post("/api/search", json={"like_tile_id": "nope"}).status_code == 404 and client.post("/api/search", json={"text": "   "}).status_code == 400
def test_tile_detail(tmp_path):
    client = client_factory(tmp_path); body = client.get("/api/tiles/32643_1_0").json()
    assert body["years_available"] == [2021, 2025] and body["change"] is not None and "google.com/maps" in body["maps_url"]
    assert client.get("/api/tiles/nope").status_code == 404
def test_change_endpoint(tmp_path):
    client = client_factory(tmp_path); body = client.get("/api/tiles/32643_1_0/change")
    assert body.status_code == 200 and len(body.json()["polygons"]["features"]) == 1 and body.json()["preview_url"].endswith("_preview.png")
    assert client.get("/api/tiles/32643_1_20/change").status_code == 404
def test_files_served(tmp_path):
    client = client_factory(tmp_path); thumb = client.post("/api/search", json={"text": "farm"}).json()["results"][0]["thumb_url"]
    assert client.get(thumb).status_code == 200 and client.get(thumb).headers["content-type"].startswith("image/jpeg")
def test_root_without_build(tmp_path): assert client_factory(tmp_path).get("/").status_code == 200
def test_reload_and_missing_data(tmp_path):
    client = TestClient(create_app(tmp_path, lambda texts: np.zeros((1, 512), dtype=np.float32)))
    assert client.get("/api/health").json()["tiles_embedded"] == 0 and client.post("/api/reload").status_code == 200
    assert client.post("/api/search", json={"text": "x"}).json()["results"] == []
def test_no_nan_in_json(tmp_path):
    assert "NaN" not in client_factory(tmp_path).post("/api/search", json={"text": "farm"}).text
