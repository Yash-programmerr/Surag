const categories = ["agriculture", "city_structures", "open_land", "urban_areas"];

export default function Filters({ value, onChange }) {
  const update = (patch) => onChange({ ...value, ...patch });
  const updateNumber = (key, raw) => update({ [key]: raw === "" ? undefined : Number(raw) / 100 });
  return <section className="filters">
    <h3>Filters</h3>
    <div className="category-list">{categories.map((category) =>
      <label key={category}><input type="checkbox" checked={value.category?.includes(category) || false}
        onChange={(event) => {
          const next = new Set(value.category || []);
          event.target.checked ? next.add(category) : next.delete(category);
          update({ category: next.size ? [...next] : undefined });
        }} /> {category.replace("_", " ")}</label>
    )}</div>
    <div className="number-grid">
      {[["min_built", "Min built %"], ["min_crop", "Min crop %"], ["min_open", "Min open %"], ["max_water", "Max water %"]].map(([key, label]) =>
        <label key={key}>{label}<input type="number" min="0" max="100" value={value[key] === undefined ? "" : value[key] * 100}
          onChange={(event) => updateNumber(key, event.target.value)} /></label>
      )}
    </div>
    <label>Results <input type="number" min="1" max="60" value={value.k}
      onChange={(event) => update({ k: Math.max(1, Math.min(60, Number(event.target.value) || 1)) })} /></label>
  </section>;
}
