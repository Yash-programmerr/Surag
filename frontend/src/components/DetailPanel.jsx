export default function DetailPanel({ detail, onSimilar, onShowChange }) {
  if (!detail) return null;
  return <section className="detail-panel">
    <h2>{detail.tile_id}</h2><img src={detail.thumb_url} alt={`Tile ${detail.tile_id}`} />
    <p>{detail.category} · {detail.lat.toFixed(4)}, {detail.lon.toFixed(4)} · EPSG:{detail.epsg}</p>
    <p>Built {(detail.built_frac * 100).toFixed(1)}% · Crop {(detail.crop_frac * 100).toFixed(1)}% · Open {(detail.open_frac * 100).toFixed(1)}% · Water {(detail.water_frac * 100).toFixed(1)}%</p>
    <div className="button-row"><button onClick={() => onSimilar(detail.tile_id)}>Find similar tiles</button>
      <a href={detail.maps_url} target="_blank" rel="noopener noreferrer">Open in Google Maps</a>
      {detail.change && <button onClick={() => onShowChange(detail.tile_id)}>Show change 2021 → 2025</button>}</div>
  </section>;
}
