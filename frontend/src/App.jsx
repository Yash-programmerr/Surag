import { useEffect, useState } from "react";
import * as api from "./api";
import SearchBar from "./components/SearchBar";
import Chips from "./components/Chips";
import Filters from "./components/Filters";
import ResultsGrid from "./components/ResultsGrid";
import DetailPanel from "./components/DetailPanel";
import ChangePanel from "./components/ChangePanel";
import MapView from "./components/MapView";

const initialFilters = { category: undefined, min_built: undefined, max_built: undefined, min_crop: undefined, min_open: undefined, max_water: undefined, k: 12 };
export default function App() {
  const [query, setQuery] = useState(""), [filters, setFilters] = useState(initialFilters);
  const [results, setResults] = useState([]), [loading, setLoading] = useState(false), [error, setError] = useState("");
  const [selectedId, setSelectedId] = useState(null), [detail, setDetail] = useState(null), [changeData, setChangeData] = useState(null);
  const [statusInfo, setStatusInfo] = useState(null), [hasSearched, setHasSearched] = useState(false);
  useEffect(() => { api.getHealth().then(setStatusInfo).catch((err) => setError(err.message)); }, []);
  const search = async (text = query, likeTileId) => {
    setLoading(true); setError(""); setHasSearched(true); setChangeData(null);
    try { const data = await api.searchTiles({ text: likeTileId ? undefined : text, like_tile_id: likeTileId, k: filters.k, filters });
      setResults(data.results); setSelectedId(null); setDetail(null);
    } catch (err) { setResults([]); setError(err.message); } finally { setLoading(false); }
  };
  const select = async (tileId) => { setSelectedId(tileId); setChangeData(null); try { setDetail(await api.getTile(tileId)); } catch (err) { setError(err.message); } };
  const showChange = async (tileId) => { try { setChangeData(await api.getChange(tileId)); } catch (err) { setError(err.message); } };
  const reload = async () => { try { setStatusInfo(await api.reloadData()); } catch (err) { setError(err.message); } };
  return <main className="app-shell"><aside className="sidebar"><header><h1>SURAG Tile Search</h1>
    {statusInfo && <small>{statusInfo.tiles_embedded} embedded · {statusInfo.tiles_with_change} with change</small>}</header>
    <SearchBar query={query} onChange={setQuery} onSearch={() => search()} loading={loading} />
    <Chips onChoose={(text) => { setQuery(text); search(text); }} /><Filters value={filters} onChange={setFilters} />
    <button className="reload" onClick={reload}>Reload data</button>
    <ResultsGrid results={results} loading={loading} error={error} hasSearched={hasSearched} onSelect={select} onSimilar={(id) => search("", id)} />
    <DetailPanel detail={detail} onSimilar={(id) => search("", id)} onShowChange={showChange} />
    <ChangePanel data={changeData} /></aside>
    <MapView results={results} selected={detail} changePolygons={changeData?.polygons} onSelect={select} /></main>;
}
