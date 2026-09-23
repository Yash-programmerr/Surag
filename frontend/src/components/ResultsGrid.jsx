import ResultCard from "./ResultCard";
export default function ResultsGrid({ results, loading, error, hasSearched, onSelect, onSimilar }) {
  if (loading) return <p>Searching...</p>;
  if (error) return <p className="error">{error}</p>;
  if (hasSearched && !results.length) return <p>No tiles match</p>;
  return <div className="results-grid">{results.map((result) =>
    <ResultCard key={result.tile_id} result={result} onSelect={onSelect} onSimilar={onSimilar} />
  )}</div>;
}
