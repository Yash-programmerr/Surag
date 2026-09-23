const colors = { agriculture: "#43a047", city_structures: "#e53935", open_land: "#8d8d8d", urban_areas: "#fb8c00" };
export default function ResultCard({ result, onSelect, onSimilar }) {
  return <article className="result-card" onClick={() => onSelect(result.tile_id)}>
    <img src={result.thumb_url} alt={`Satellite tile ${result.tile_id}`} />
    <div><strong>#{result.rank}</strong> <span>{result.score.toFixed(3)}</span></div>
    <div><i className="category-dot" style={{ background: colors[result.category] }} /> {result.category}</div>
    <small>built {(result.built_frac * 100).toFixed(0)}% · crop {(result.crop_frac * 100).toFixed(0)}% · open {(result.open_frac * 100).toFixed(0)}%</small>
    <button type="button" onClick={(event) => { event.stopPropagation(); onSimilar(result.tile_id); }}>Similar</button>
  </article>;
}
