const examples = ["dense buildings in a city", "farmland with crop fields", "barren open land",
  "suburban residential area", "river or water body", "road network", "new construction site"];

export default function Chips({ onChoose }) {
  return <div className="chips">{examples.map((example) =>
    <button type="button" key={example} onClick={() => onChoose(example)}>{example}</button>
  )}</div>;
}
