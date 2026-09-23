#!/usr/bin/env python3
"""
Bulk satellite tile pipeline for India (Google Earth Engine -> local 256x256 GeoTIFF tiles)
==========================================================================================
Two stages, both resumable:

  python bulk_tiles.py select   [--quota-scale 1.0]   # pick tile locations per category
  python bulk_tiles.py download  [--workers 8] [--limit N]
  python bulk_tiles.py status

Stage 1 (select)
  - Random points inside India, stratified by GHSL settlement class (rural / suburban /
    urban cluster / city centre).
  - Every point maps to a cell of a fixed UTM tile grid (2560 m = 256 px @ 10 m), so tiles
    never overlap and tile_id is stable:  <EPSG>_<i>_<j>
  - ESA WorldCover 2021 class fractions (cropland / built / grass+bare / water) are computed for
    each candidate tile and rules assign it to a category:
        agriculture, city_structures, open_land, urban_areas
  - Stops when the per-category QUOTAS are met (default total = 20,000 tiles).

Stage 2 (download)
  - One 256x256 GeoTIFF per tile per year via ee.data.computePixels on the high-volume endpoint
    (no Drive, no manual tasks). Parallel threads, retries, resume (existing files are skipped).
  - MAIN_YEAR for every tile; ~10% of tiles (CHANGE_FRACTION) also get CHANGE_YEAR
    -> these are your change-detection pairs.
  - Bands (int16, nodata -9999): B2,B3,B4,B8,B11,B12 (x10000 reflectance) + WorldCover_2021 label.
  - Also writes an RGB JPEG thumbnail per tile (for CLIP embeddings / the UI).

Output:
  OUT_DIR/accepted_tiles.csv   selected tiles + category + land-cover fractions + smod code
  OUT_DIR/tiles_index.csv      one row per downloaded tile-year (paths, valid_fraction, ndvi/ndbi means, ...)
  OUT_DIR/failures.csv         tile-years that failed (re-run download to retry)
  OUT_DIR/tiles/<category>/<tile_id>_<year>.tif
  OUT_DIR/thumbs/<tile_id>.jpg

Setup:
  pip install earthengine-api rasterio numpy pillow pyproj
  earthengine authenticate
  set GEE_PROJECT below
"""

import argparse
import csv
import io
import math
import random
import sys
import threading
import time
import zlib
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import ee
import numpy as np
import rasterio
from PIL import Image
from pyproj import Transformer

# ============================== CONFIG ==============================
GEE_PROJECT = "geesatellite"      # <-- CHANGE THIS
OUT_DIR = Path("india_tiles")

TILE_PX = 256
SCALE = 10
TILE_M = TILE_PX * SCALE                 # 2560 m

# target number of tiles per category (total 20,000). Cities are naturally limited;
# the script reports any shortfall instead of looping forever.
QUOTAS = {"city_structures": 3000, "urban_areas": 5000, "open_land": 6000, "agriculture": 6000}
SEED = 42

MAIN_YEAR = 2025
CHANGE_YEAR = 2021                       # second date for the change-detection subset
CHANGE_FRACTION = 0.10                   # share of tiles that also get CHANGE_YEAR
START_MD, END_MD = "01-01", "12-31"      # same window every year (dry season e.g. 02-01..04-30)

CLOUD_SCORE_MIN = 0.60
MAX_SCENE_CLOUD = 70
BANDS = ["B2", "B3", "B4", "B8", "B11", "B12"]   # order matters (index 2 = B4, 3 = B8, 4 = B11)
NODATA = -9999

WORKERS_DEFAULT = 8
ROUND_POINTS = 1500                      # sampled points per category per selection round
MAX_ROUNDS = 40
BATCH = 250                              # tiles per land-cover-fraction request

THRESH = {
    "agri_min_crop": 0.70, "agri_max_built": 0.05,
    "city_min_built": 0.35,
    "urban_min_built": 0.10, "urban_max_built": 0.40,
    "open_min_open": 0.70, "open_max_built": 0.03, "open_max_crop": 0.15,
    "max_water": 0.05,
}
# ====================================================================

T = THRESH
CAT_ORDER = ["city_structures", "urban_areas", "open_land", "agriculture"]   # rarest first
CAT_SMOD = {
    "city_structures": [30],
    "urban_areas": [21, 22, 23],
    "open_land": [11, 12],
    "agriculture": [11, 12],
}

S2 = "COPERNICUS/S2_SR_HARMONIZED"
CSP = "GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED"

ACCEPTED = OUT_DIR / "accepted_tiles.csv"
EVALUATED = OUT_DIR / "evaluated_tiles.txt"
INDEX = OUT_DIR / "tiles_index.csv"
FAILS = OUT_DIR / "failures.csv"
TILES_DIR = OUT_DIR / "tiles"
THUMBS_DIR = OUT_DIR / "thumbs"

ACCEPTED_FIELDS = ["tile_id", "category", "lon", "lat", "epsg", "x0", "y0", "x1", "y1",
                   "smod_code", "crop_frac", "built_frac", "open_frac", "water_frac"]
INDEX_FIELDS = ["tile_id", "category", "year", "path", "thumb", "lon", "lat", "epsg",
                "x0", "y0", "x1", "y1", "smod_code", "crop_frac", "built_frac", "open_frac",
                "water_frac", "valid_fraction", "ndvi_mean", "ndbi_mean", "n_scenes",
                "mean_scene_cloud_pct", "bands", "downloaded_utc"]

_lock = threading.Lock()
_transformers = {}


# ------------------------- init -------------------------
def ee_init():
    for kwargs in (
        {"project": GEE_PROJECT, "opt_url": "https://earthengine-highvolume.googleapis.com"},
        {"project": GEE_PROJECT},
    ):
        try:
            ee.Initialize(**kwargs)
            return
        except Exception:  # noqa: BLE001
            continue
    ee.Authenticate()
    ee.Initialize(project=GEE_PROJECT, opt_url="https://earthengine-highvolume.googleapis.com")


def with_retries(fn, tries=5):
    for i in range(tries):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001
            msg = str(e).lower()
            if "did not match" in msg or "no bands" in msg:
                raise RuntimeError("no_scenes_for_tile") from e
            if i == tries - 1:
                raise
            time.sleep(min(60, 3 * 2 ** i) + random.random())


# ------------------------- tile grid -------------------------
def epsg_for(lon, lat):
    zone = int((lon + 180) // 6) + 1
    return (32600 if lat >= 0 else 32700) + zone


def tile_of(lon, lat):
    epsg = epsg_for(lon, lat)
    if epsg not in _transformers:
        _transformers[epsg] = Transformer.from_crs(4326, epsg, always_xy=True)
    x, y = _transformers[epsg].transform(lon, lat)
    return epsg, math.floor(x / TILE_M), math.floor(y / TILE_M)


# ------------------------- csv helpers -------------------------
def read_csv(path):
    if not path.exists():
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def append_csv(path, fields, rows):
    with _lock:
        new = not path.exists()
        with open(path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            if new:
                w.writeheader()
            w.writerows(rows)


# ------------------------- GEE layers (built lazily) -------------------------
_layers = {}


def layers():
    if not _layers:
        _layers["india"] = (ee.FeatureCollection("USDOS/LSIB_SIMPLE/2017")
                            .filter(ee.Filter.eq("country_na", "India")).geometry())
        _layers["smod"] = ee.Image("JRC/GHSL/P2023A/GHS_SMOD/2020").select("smod_code")
        wc = ee.ImageCollection("ESA/WorldCover/v200").first().select("Map")
        _layers["wc"] = wc
        _layers["frac"] = ee.Image.cat([
            wc.eq(40).rename("crop"),
            wc.eq(50).rename("built"),
            wc.eq(30).Or(wc.eq(60)).rename("open"),
            wc.eq(80).rename("water"),
        ])
        _layers["label"] = wc.rename("WorldCover_2021").toInt16().unmask(0)
    return _layers


# ============================ STAGE 1: SELECT ============================
def sample_points(smod_classes, n_total, seed):
    L = layers()
    per = max(1, n_total // len(smod_classes))
    fc = L["smod"].stratifiedSample(
        numPoints=0, classBand="smod_code", region=L["india"], scale=1000, seed=seed,
        classValues=smod_classes, classPoints=[per] * len(smod_classes),
        geometries=True, tileScale=4)
    feats = with_retries(lambda: fc.getInfo())["features"]
    return [(f["geometry"]["coordinates"][0], f["geometry"]["coordinates"][1],
             int(f["properties"]["smod_code"])) for f in feats]


def eval_fractions(batch):
    L = layers()
    feats = []
    for it in batch:
        rect = ee.Geometry.Rectangle([it["x0"], it["y0"], it["x1"], it["y1"]],
                                     proj=f"EPSG:{it['epsg']}", geodesic=False)
        feats.append(ee.Feature(rect, {"tile_id": it["tile_id"]}))
    fc = L["frac"].reduceRegions(collection=ee.FeatureCollection(feats),
                                 reducer=ee.Reducer.mean(), scale=40, tileScale=4)
    fc = fc.select(["tile_id", "crop", "built", "open", "water"], None, False)
    res = with_retries(lambda: fc.getInfo())["features"]
    return {f["properties"]["tile_id"]: f["properties"] for f in res}


def passes(cat, c):
    if cat == "city_structures":
        return c["built"] >= T["city_min_built"] and c["water"] <= T["max_water"]
    if cat == "urban_areas":
        return (T["urban_min_built"] <= c["built"] <= T["urban_max_built"]
                and c["water"] <= T["max_water"])
    if cat == "open_land":
        return (c["open"] >= T["open_min_open"] and c["built"] <= T["open_max_built"]
                and c["crop"] <= T["open_max_crop"] and c["water"] <= T["max_water"])
    if cat == "agriculture":
        return (c["crop"] >= T["agri_min_crop"] and c["built"] <= T["agri_max_built"]
                and c["water"] <= T["max_water"])
    return False


def select(quota_scale):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ee_init()
    quotas = {k: max(1, int(v * quota_scale)) for k, v in QUOTAS.items()}
    accepted = {r["tile_id"]: r for r in read_csv(ACCEPTED)}
    counts = Counter(r["category"] for r in accepted.values())
    evaluated = set(EVALUATED.read_text().split()) if EVALUATED.exists() else set()
    print(f"Resuming with {len(accepted)} accepted tiles, {len(evaluated)} already evaluated.")
    print("Quotas:", quotas)

    for rnd in range(1, MAX_ROUNDS + 1):
        need = [c for c in CAT_ORDER if counts[c] < quotas[c]]
        if not need:
            break
        # unique random offset per run so a resumed run doesn't resample the same points
        run_salt = int(time.time()) % 100000
        cands = {}
        for cat in need:
            seed = SEED + run_salt + rnd * 100 + CAT_ORDER.index(cat)
            try:
                pts = sample_points(CAT_SMOD[cat], ROUND_POINTS, seed)
            except Exception as e:  # noqa: BLE001
                print(f"  sampling failed for {cat}: {e}")
                continue
            for lon, lat, sm in pts:
                epsg, i, j = tile_of(lon, lat)
                tid = f"{epsg}_{i}_{j}"
                if tid in evaluated or tid in accepted or tid in cands:
                    continue
                cands[tid] = {"tile_id": tid, "lon": lon, "lat": lat, "epsg": epsg,
                              "x0": i * TILE_M, "y0": j * TILE_M,
                              "x1": (i + 1) * TILE_M, "y1": (j + 1) * TILE_M, "smod_code": sm}
        items = list(cands.values())
        print(f"Round {rnd}: {len(items)} new candidate tiles | need: "
              + ", ".join(f"{c} {counts[c]}/{quotas[c]}" for c in need))
        if not items:
            print("  no new candidates found; stopping.")
            break

        for b in range(0, len(items), BATCH):
            batch = items[b:b + BATCH]
            try:
                fr = eval_fractions(batch)
            except Exception as e:  # noqa: BLE001
                print(f"  fraction batch failed ({e}); skipping batch")
                continue
            new_rows = []
            for it in batch:
                evaluated.add(it["tile_id"])
                c = fr.get(it["tile_id"])
                if not c or any(c.get(k) is None for k in ("crop", "built", "open", "water")):
                    continue
                for cat in CAT_ORDER:
                    if counts[cat] >= quotas[cat] or it["smod_code"] not in CAT_SMOD[cat]:
                        continue
                    if passes(cat, c):
                        row = dict(it, category=cat, crop_frac=round(c["crop"], 4),
                                   built_frac=round(c["built"], 4), open_frac=round(c["open"], 4),
                                   water_frac=round(c["water"], 4))
                        accepted[it["tile_id"]] = row
                        counts[cat] += 1
                        new_rows.append(row)
                        break
            if new_rows:
                append_csv(ACCEPTED, ACCEPTED_FIELDS, new_rows)
            with open(EVALUATED, "a") as f:
                f.write("\n".join(it["tile_id"] for it in batch) + "\n")
            print(f"  batch {b // BATCH + 1}/{math.ceil(len(items) / BATCH)}: "
                  + ", ".join(f"{c}={counts[c]}" for c in CAT_ORDER))

    print("\nSelection finished:")
    for c in CAT_ORDER:
        flag = "" if counts[c] >= quotas[c] else "   <-- SHORTFALL (loosen THRESH or raise MAX_ROUNDS)"
        print(f"  {c:16s} {counts[c]:5d} / {quotas[c]}{flag}")
    print(f"Total accepted: {sum(counts.values())}  ->  {ACCEPTED}")


# ============================ STAGE 2: DOWNLOAD ============================
def is_change_tile(tile_id):
    return (zlib.crc32(tile_id.encode()) % 1000) < int(CHANGE_FRACTION * 1000)


def years_for(row):
    return [MAIN_YEAR] + ([CHANGE_YEAR] if is_change_tile(row["tile_id"]) else [])


def tile_path(row, year):
    return TILES_DIR / row["category"] / f"{row['tile_id']}_{year}.tif"


def build_image(row, year):
    L = layers()
    region = ee.Geometry.Rectangle(
        [float(row["x0"]), float(row["y0"]), float(row["x1"]), float(row["y1"])],
        proj=f"EPSG:{row['epsg']}", geodesic=False)
    start = f"{year}-{START_MD}"
    end = ee.Date(f"{year}-{END_MD}").advance(1, "day")
    col = (ee.ImageCollection(S2).filterBounds(region).filterDate(start, end)
           .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", MAX_SCENE_CLOUD))
           .linkCollection(ee.ImageCollection(CSP), ["cs_cdf"]))
    masked = col.map(lambda im: im.updateMask(im.select("cs_cdf").gte(CLOUD_SCORE_MIN)))
    comp = masked.select(BANDS).median()
    optical = comp.round().toInt16().unmask(NODATA).toInt16()
    return optical.addBands(L["label"]), col


def make_thumb(arr, dest):
    rgb = arr[[2, 1, 0]].astype(np.float32)
    valid = (arr[:6] != NODATA).all(axis=0)
    rgb = np.clip(rgb / 3000.0, 0, 1) ** 0.8
    rgb[:, ~valid] = 0
    Image.fromarray((rgb.transpose(1, 2, 0) * 255).astype(np.uint8)).save(dest, quality=90)


def fetch_tile(row, year, scene_info):
    dest = tile_path(row, year)
    dest.parent.mkdir(parents=True, exist_ok=True)
    x0, y1 = float(row["x0"]), float(row["y1"])
    image, col = build_image(row, year)
    req = {
        "expression": image,
        "fileFormat": "GEO_TIFF",
        "grid": {
            "dimensions": {"width": TILE_PX, "height": TILE_PX},
            "affineTransform": {"scaleX": SCALE, "shearX": 0, "translateX": x0,
                                "shearY": 0, "scaleY": -SCALE, "translateY": y1},
            "crsCode": f"EPSG:{row['epsg']}",
        },
    }
    data = with_retries(lambda: ee.data.computePixels(req))

    with rasterio.MemoryFile(data) as mf:
        with mf.open() as ds:
            arr = ds.read()
    if arr.shape != (len(BANDS) + 1, TILE_PX, TILE_PX):
        raise RuntimeError(f"unexpected tile shape {arr.shape}")

    valid = (arr[:6] != NODATA).all(axis=0)
    vf = float(valid.mean())
    ndvi = ndbi = None
    if valid.any():
        b4, b8, b11 = (arr[i][valid].astype(np.float32) for i in (2, 3, 4))
        ndvi = float(np.mean((b8 - b4) / np.maximum(b8 + b4, 1)))
        ndbi = float(np.mean((b11 - b8) / np.maximum(b11 + b8, 1)))

    tmp = dest.with_suffix(".tmp")
    tmp.write_bytes(data)
    tmp.replace(dest)

    thumb = ""
    if year == MAIN_YEAR:
        THUMBS_DIR.mkdir(parents=True, exist_ok=True)
        thumb_path = THUMBS_DIR / f"{row['tile_id']}.jpg"
        make_thumb(arr, thumb_path)
        thumb = str(thumb_path)

    n_scenes = cloud = ""
    if scene_info:
        info = with_retries(lambda: ee.Dictionary({
            "n": col.size(), "c": col.aggregate_mean("CLOUDY_PIXEL_PERCENTAGE")}).getInfo())
        n_scenes, cloud = info.get("n"), info.get("c")

    return dict(row, year=year, path=str(dest), thumb=thumb,
                valid_fraction=round(vf, 4),
                ndvi_mean=None if ndvi is None else round(ndvi, 4),
                ndbi_mean=None if ndbi is None else round(ndbi, 4),
                n_scenes=n_scenes, mean_scene_cloud_pct=cloud,
                bands=",".join(BANDS + ["WorldCover_2021"]),
                downloaded_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"))


def download(workers, limit, scene_info):
    ee_init()
    rows = read_csv(ACCEPTED)
    if not rows:
        sys.exit("No accepted tiles. Run:  python bulk_tiles.py select")
    jobs = [(r, y) for r in rows for y in years_for(r) if not tile_path(r, y).exists()]
    random.Random(SEED).shuffle(jobs)          # mixed categories early, useful if you stop midway
    if limit:
        jobs = jobs[:limit]
    print(f"Tiles selected: {len(rows)} | tile-years still to download: {len(jobs)} | workers: {workers}")
    if not jobs:
        return

    done = failed = 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(fetch_tile, r, y, scene_info): (r, y) for r, y in jobs}
        for fut in as_completed(futs):
            r, y = futs[fut]
            try:
                append_csv(INDEX, INDEX_FIELDS, [fut.result()])
                done += 1
            except Exception as e:  # noqa: BLE001
                failed += 1
                append_csv(FAILS, ["tile_id", "year", "error"],
                           [{"tile_id": r["tile_id"], "year": y, "error": str(e)[:300]}])
            n = done + failed
            if n % 50 == 0 or n == len(jobs):
                rate = n / max(time.time() - t0, 1)
                eta = (len(jobs) - n) / max(rate, 1e-6) / 60
                print(f"  {n}/{len(jobs)} done ({failed} failed) | {rate:.2f} tiles/s | ETA {eta:.0f} min")
    print(f"Finished. ok={done} failed={failed}. Re-run 'download' to retry failures.")


def status():
    acc = read_csv(ACCEPTED)
    idx = read_csv(INDEX)
    fails = read_csv(FAILS)
    print(f"Accepted tiles: {len(acc)}  " + str(dict(Counter(r['category'] for r in acc))))
    print(f"Downloaded tile-years: {len(idx)}  by year: {dict(Counter(r['year'] for r in idx))}")
    low = sum(1 for r in idx if float(r["valid_fraction"] or 0) < 0.9)
    print(f"Tiles with <90% valid pixels (clouds/no data): {low}")
    print(f"Failures logged: {len(fails)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("select")
    s.add_argument("--quota-scale", type=float, default=1.0, help="e.g. 0.01 for a tiny test run")
    d = sub.add_parser("download")
    d.add_argument("--workers", type=int, default=WORKERS_DEFAULT)
    d.add_argument("--limit", type=int, default=0, help="only download N tile-years (testing)")
    d.add_argument("--scene-info", action="store_true", help="also record n_scenes / mean cloud %% (extra request per tile)")
    sub.add_parser("status")
    a = ap.parse_args()

    if a.cmd == "select":
        select(a.quota_scale)
    elif a.cmd == "download":
        download(a.workers, a.limit, a.scene_info)
    else:
        status()


if __name__ == "__main__":
    main()
