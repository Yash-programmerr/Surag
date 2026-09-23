const classes = [["#ff0000", "New built"], ["#ff8c00", "Vegetation loss"], ["#00c800", "Vegetation gain"], ["#0064ff", "Water change"], ["#ffeb00", "Other change"]];
export default function Legend() {
  return <div className="legend">{classes.map(([color, label]) => <div key={label}><i style={{ background: color }} />{label}</div>)}</div>;
}
