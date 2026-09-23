export default function ChangePanel({ data }) {
  if (!data) return null;
  const s = data.summary;
  return <section className="change-panel"><h2>Change 2021 → 2025</h2>
    <img src={data.preview_url} alt="2021, 2025, and change preview" />
    <p className="caption">Left: 2021 | Middle: 2025 | Right: changes</p>
    <table><tbody>{[["New built ha", s.new_built_ha], ["Vegetation loss ha", s.vegetation_loss_ha],
      ["Vegetation gain ha", s.vegetation_gain_ha], ["Water change ha", s.water_change_ha],
      ["Other ha", s.other_change_ha], ["Changed % of tile", `${(s.changed_fraction * 100).toFixed(2)}%`]].map(([label, value]) =>
      <tr key={label}><th>{label}</th><td>{typeof value === "number" ? value.toFixed(2) : value}</td></tr>)}</tbody></table>
    {s.suspicious && <p className="warning">Many pixels changed - possibly seasonal or cloud effects</p>}
  </section>;
}
