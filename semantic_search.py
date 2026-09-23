#!/usr/bin/env python3
"""
Semantic search over your downloaded satellite tiles (MVP)
==========================================================
Uses CLIP on the RGB thumbnails written by bulk_tiles.py (india_tiles/thumbs/*.jpg).
No FAISS needed: 20k x 512 embeddings search instantly with a numpy matrix product.

  python3 semantic_search.py embed                      # embed all new thumbnails (resumable, incremental)
  python3 semantic_search.py embed --limit 200          # quick test on 200 tiles
  python3 semantic_search.py search --text "dense buildings in a city" --k 12
  python3 semantic_search.py search --text "farmland" --category agriculture --min-crop 0.8
  python3 semantic_search.py search --like 32643_130_1080 --k 12    # tiles similar to a given tile_id
  python3 semantic_search.py eval                       # precision@k using your category labels

`search` prints a table and opens results.html (thumbnails + Google Maps satellite links).

Install:
  pip install torch transformers pillow numpy certifi
  (first run downloads the CLIP model, ~600 MB)
"""

import argparse
import csv
import html
import json
import sys
import webbrowser
from pathlib import Path

import numpy as np
from PIL import Image

OUT_DIR = Path("india_tiles")
INDEX_CSV = OUT_DIR / "tiles_index.csv"
EMB_DIR = OUT_DIR / "embeddings"
EMB_FILE = EMB_DIR / "embeddings.npy"
IDS_FILE = EMB_DIR / "tile_ids.json"
RESULTS_HTML = Path("results.html")

MODEL_NAME = "openai/clip-vit-base-patch32"
BATCH = 64
SAVE_EVERY_BATCHES = 20
TEXT_TEMPLATE = "a satellite image of {}"

PROMPTS = {   # used by `eval` (text -> image) with category as ground truth
    "agriculture": ["farmland with crop fields", "agricultural fields seen from space", "green cultivated land"],
    "city_structures": ["dense buildings in a city", "urban core with many buildings and roads",
                        "high density city blocks"],
    "open_land": ["barren open land", "empty bare ground and grassland", "vacant open land with sparse vegetation"],
    "urban_areas": ["suburban residential area", "town with houses and some vegetation",
                    "urban fringe with scattered buildings"],
}

FLOAT_FIELDS = ["crop_frac", "built_frac", "open_frac", "water_frac", "valid_fraction",
                "ndvi_mean", "ndbi_mean", "lon", "lat"]


# ------------------------- model -------------------------
_model = {}


def get_model():
    if not _model:
        import torch # type: ignore
        from transformers import CLIPModel, CLIPProcessor # type: ignore
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
        print(f"Loading {MODEL_NAME} on {device} ...")
        _model.update(
            torch=torch, device=device,
            model=CLIPModel.from_pretrained(MODEL_NAME).to(device).eval(),
            proc=CLIPProcessor.from_pretrained(MODEL_NAME))
    return _model


def _as_tensor(out):
    torch = _model["torch"]
    if torch.is_tensor(out):
        return out
    if hasattr(out, "pooler_output") and out.pooler_output is not None:
        return out.pooler_output
    return out[0]


def embed_images(paths):
    m = get_model()
    torch = m["torch"]
    imgs = [Image.open(p).convert("RGB") for p in paths]
    inputs = m["proc"](images=imgs, return_tensors="pt").to(m["device"])
    with torch.no_grad():
        f = _as_tensor(m["model"].get_image_features(**inputs))
        f = torch.nn.functional.normalize(f, dim=-1)
    return f.cpu().numpy().astype(np.float32)


def embed_texts(texts):
    m = get_model()
    torch = m["torch"]
    inputs = m["proc"](text=texts, return_tensors="pt", padding=True).to(m["device"])
    with torch.no_grad():
        f = _as_tensor(m["model"].get_text_features(**inputs))
        f = torch.nn.functional.normalize(f, dim=-1)
    return f.cpu().numpy().astype(np.float32)


# ------------------------- data -------------------------
def load_meta():
    """tile_id -> metadata dict, for tiles that have a thumbnail (main-year rows)."""
    if not INDEX_CSV.exists():
        sys.exit(f"{INDEX_CSV} not found. Run bulk_tiles.py download first.")
    meta = {}
    with open(INDEX_CSV, newline="") as f:
        for r in csv.DictReader(f):
            if not r.get("thumb") or not Path(r["thumb"]).exists():
                continue
            for k in FLOAT_FIELDS:
                try:
                    r[k] = float(r[k])
                except (TypeError, ValueError):
                    r[k] = float("nan")
            meta[r["tile_id"]] = r
    return meta


def save_store(ids, emb):
    EMB_DIR.mkdir(parents=True, exist_ok=True)
    tmp_npy = EMB_FILE.with_name("embeddings.tmp.npy")
    np.save(tmp_npy, emb)
    tmp_npy.replace(EMB_FILE)
    tmp_ids = IDS_FILE.with_suffix(".tmp")
    tmp_ids.write_text(json.dumps(ids))
    tmp_ids.replace(IDS_FILE)


def load_store():
    if not EMB_FILE.exists():
        sys.exit("No embeddings yet. Run:  python3 semantic_search.py embed")
    emb = np.load(EMB_FILE)
    ids = json.loads(IDS_FILE.read_text())
    meta = load_meta()
    keep = [i for i, t in enumerate(ids) if t in meta]
    return emb[keep], [ids[i] for i in keep], meta


# ------------------------- embed -------------------------
def cmd_embed(args):
    meta = load_meta()
    ids, emb = [], np.zeros((0, 512), dtype=np.float32)
    if EMB_FILE.exists() and IDS_FILE.exists():
        emb = np.load(EMB_FILE)
        ids = json.loads(IDS_FILE.read_text())
    done = set(ids)
    todo = [t for t in meta if t not in done]
    if args.limit:
        todo = todo[:args.limit]
    print(f"Thumbnails available: {len(meta)} | already embedded: {len(done)} | to embed now: {len(todo)}")
    if not todo:
        return

    new_ids, new_vecs = [], []
    n_batches = (len(todo) + BATCH - 1) // BATCH
    for bi in range(n_batches):
        batch = todo[bi * BATCH:(bi + 1) * BATCH]
        try:
            vecs = embed_images([meta[t]["thumb"] for t in batch])
        except Exception as e:  # noqa: BLE001
            print(f"  batch {bi + 1}: failed as a group ({e}); retrying one by one")
            vecs, ok = [], []
            for t in batch:
                try:
                    vecs.append(embed_images([meta[t]["thumb"]])[0])
                    ok.append(t)
                except Exception as e2:  # noqa: BLE001
                    print(f"    skipping {t}: {e2}")
            batch, vecs = ok, (np.stack(vecs) if vecs else np.zeros((0, emb.shape[1]), np.float32))
        new_ids += batch
        new_vecs.append(vecs)
        if (bi + 1) % 10 == 0 or bi + 1 == n_batches:
            print(f"  {min((bi + 1) * BATCH, len(todo))}/{len(todo)} embedded")
        if (bi + 1) % SAVE_EVERY_BATCHES == 0 or bi + 1 == n_batches:
            all_vecs = np.concatenate([emb] + new_vecs) if new_vecs else emb
            save_store(ids + new_ids, all_vecs)
    print(f"Saved {len(ids) + len(new_ids)} embeddings -> {EMB_FILE}")


# ------------------------- search -------------------------
def build_mask(ids, meta, a):
    m = np.ones(len(ids), dtype=bool)
    cats = set(a.category) if a.category else None
    for i, t in enumerate(ids):
        r = meta[t]
        if cats and r["category"] not in cats:
            m[i] = False
        elif a.min_built is not None and not r["built_frac"] >= a.min_built:
            m[i] = False
        elif a.max_built is not None and not r["built_frac"] <= a.max_built:
            m[i] = False
        elif a.min_crop is not None and not r["crop_frac"] >= a.min_crop:
            m[i] = False
        elif a.min_open is not None and not r["open_frac"] >= a.min_open:
            m[i] = False
        elif a.max_water is not None and not r["water_frac"] <= a.max_water:
            m[i] = False
        elif not r["valid_fraction"] >= a.min_valid:
            m[i] = False
    return m


def maps_link(r):
    return f"https://www.google.com/maps/@{r['lat']},{r['lon']},14z/data=!3m1!1e3"


def write_gallery(hits, title):
    cards = []
    for rank, (t, score, r) in enumerate(hits, 1):
        cards.append(
            f"<div class='c'><img src='{Path(r['thumb']).resolve().as_uri()}'>"
            f"<div><b>#{rank}</b> score {score:.3f}<br>{html.escape(r['category'])}<br>"
            f"built {r['built_frac']:.2f} | crop {r['crop_frac']:.2f} | open {r['open_frac']:.2f}<br>"
            f"{r['lat']:.4f}, {r['lon']:.4f}<br><a href='{maps_link(r)}' target='_blank'>open in Maps</a>"
            f"<br><small>{t}</small></div></div>")
    RESULTS_HTML.write_text(
        "<html><head><meta charset='utf-8'><style>body{font-family:sans-serif;margin:16px}"
        ".g{display:grid;grid-template-columns:repeat(auto-fill,minmax(230px,1fr));gap:12px}"
        ".c{border:1px solid #ccc;border-radius:6px;padding:6px;font-size:12px}"
        ".c img{width:100%;border-radius:4px}</style></head><body>"
        f"<h2>{html.escape(title)}</h2><div class='g'>{''.join(cards)}</div></body></html>")


def cmd_search(args):
    emb, ids, meta = load_store()
    if bool(args.text) == bool(args.like):
        sys.exit("Give exactly one of --text or --like")

    if args.text:
        q = embed_texts([TEXT_TEMPLATE.format(args.text)])[0]
        title = f"Text query: {args.text}"
        self_idx = None
    else:
        if args.like not in ids:
            sys.exit(f"tile_id {args.like} not embedded yet")
        self_idx = ids.index(args.like)
        q = emb[self_idx]
        title = f"Similar to {args.like}"

    scores = emb @ q
    scores[~build_mask(ids, meta, args)] = -np.inf
    if self_idx is not None:
        scores[self_idx] = -np.inf
    k = min(args.k, int(np.isfinite(scores).sum()))
    if k == 0:
        sys.exit("No tiles match the filters.")
    top = np.argpartition(-scores, k - 1)[:k]
    top = top[np.argsort(-scores[top])]

    hits = [(ids[i], float(scores[i]), meta[ids[i]]) for i in top]
    print(f"\n{title}   (searching {len(ids)} tiles)\n")
    print(f"{'#':>3} {'score':>6}  {'category':16} {'lat':>8} {'lon':>8}  built  crop  open  tile_id")
    for rank, (t, s, r) in enumerate(hits, 1):
        print(f"{rank:>3} {s:6.3f}  {r['category']:16} {r['lat']:8.3f} {r['lon']:8.3f}  "
              f"{r['built_frac']:.2f}  {r['crop_frac']:.2f}  {r['open_frac']:.2f}  {t}")
    write_gallery(hits, title)
    print(f"\nGallery written to {RESULTS_HTML.resolve()}")
    if not args.no_open:
        webbrowser.open(RESULTS_HTML.resolve().as_uri())


# ------------------------- eval -------------------------
def cmd_eval(args):
    emb, ids, meta = load_store()
    n = len(ids)
    if n < 200:
        print(f"WARNING: only {n} embedded tiles; numbers will be noisy. Embed more first.")
    cats = np.array([meta[t]["category"] for t in ids])
    names = sorted(set(cats))
    priors = np.array([(cats == c).mean() for c in names])
    baseline = float((priors ** 2).sum())
    K = args.k

    rng = np.random.default_rng(0)
    q_idx = rng.choice(n, size=min(args.n, n), replace=False)
    sims = emb[q_idx] @ emb.T
    sims[np.arange(len(q_idx)), q_idx] = -np.inf
    top = np.argsort(-sims, axis=1)[:, :K]
    hit = cats[top] == cats[q_idx][:, None]

    print(f"\n=== Image -> image: precision@{K} (neighbour has same category as the query tile) ===")
    print(f"queries: {len(q_idx)} | tiles: {n} | random baseline: {baseline:.3f}")
    for c in names:
        m = cats[q_idx] == c
        if m.any():
            print(f"  {c:16s} {hit[m].mean():.3f}   ({m.sum()} queries)")
    print(f"  {'OVERALL':16s} {hit.mean():.3f}")

    print(f"\n=== Text -> image: precision@{K} (retrieved tile belongs to the intended category) ===")
    allp = []
    for c, prompts in PROMPTS.items():
        if c not in names:
            continue
        ps = []
        for p in prompts:
            q = embed_texts([TEXT_TEMPLATE.format(p)])[0]
            top_i = np.argsort(-(emb @ q))[:K]
            ps.append(float((cats[top_i] == c).mean()))
            print(f"  [{c:15s}] {p:45s} {ps[-1]:.3f}")
        allp += ps
    if allp:
        print(f"  {'MEAN over prompts':64s} {np.mean(allp):.3f}")
    print("\nNote: labels come from WorldCover thresholds, so agriculture vs open_land and "
          "urban_areas vs city_structures are naturally confusable.")


# ------------------------- main -------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("embed")
    e.add_argument("--limit", type=int, default=0)

    s = sub.add_parser("search")
    s.add_argument("--text")
    s.add_argument("--like", help="tile_id to find similar tiles for")
    s.add_argument("--k", type=int, default=12)
    s.add_argument("--category", nargs="+",
                   choices=["agriculture", "city_structures", "open_land", "urban_areas"])
    s.add_argument("--min-built", type=float)
    s.add_argument("--max-built", type=float)
    s.add_argument("--min-crop", type=float)
    s.add_argument("--min-open", type=float)
    s.add_argument("--max-water", type=float)
    s.add_argument("--min-valid", type=float, default=0.9, help="min share of valid (non-cloud) pixels")
    s.add_argument("--no-open", action="store_true", help="don't open the browser")

    v = sub.add_parser("eval")
    v.add_argument("--k", type=int, default=10)
    v.add_argument("--n", type=int, default=500, help="number of query tiles for image->image eval")

    a = ap.parse_args()
    {"embed": cmd_embed, "search": cmd_search, "eval": cmd_eval}[a.cmd](a)


if __name__ == "__main__":
    main()
