export default function SearchBar({ query, onChange, onSearch, loading }) {
  return <form className="search-bar" onSubmit={(event) => { event.preventDefault(); onSearch(); }}>
    <input value={query} onChange={(event) => onChange(event.target.value)} placeholder="Search satellite tiles..." />
    <button type="submit" disabled={loading}>{loading ? "Searching..." : "Search"}</button>
  </form>;
}
