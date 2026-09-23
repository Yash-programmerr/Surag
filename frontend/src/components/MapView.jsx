import { useEffect } from "react";
import { MapContainer, TileLayer, LayersControl, Rectangle, Tooltip, GeoJSON, Popup, useMap } from "react-leaflet";
import Legend from "./Legend";

const categoryColors = { city_structures: "#e53935", urban_areas: "#fb8c00", open_land: "#8d8d8d", agriculture: "#43a047" };
const changeColors = { 1: "#ff0000", 2: "#ff8c00", 3: "#00c800", 4: "#0064ff", 5: "#ffeb00" };
function FitBounds({ results, selected }) {
  const map = useMap();
  useEffect(() => {
    const bounds = selected?.bounds || results.flatMap((result) => result.bounds || []);
    if (bounds.length) map.fitBounds(bounds, { maxZoom: selected ? 14 : 12 });
  }, [map, results, selected]);
  return null;
}
export default function MapView({ results, selected, changePolygons, onSelect }) {
  return <div className="map-wrap"><MapContainer center={[22.5, 79]} zoom={5} scrollWheelZoom>
    <LayersControl position="topright"><LayersControl.BaseLayer checked name="Satellite">
      <TileLayer url="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}" attribution="Tiles &copy; Esri" />
    </LayersControl.BaseLayer><LayersControl.BaseLayer name="Streets">
      <TileLayer url="https://tile.openstreetmap.org/{z}/{x}/{y}.png" attribution="&copy; OpenStreetMap contributors" />
    </LayersControl.BaseLayer></LayersControl>
    <FitBounds results={results} selected={selected} />
    {results.map((result) => <Rectangle key={result.tile_id} bounds={result.bounds} pathOptions={{
      color: categoryColors[result.category], weight: selected?.tile_id === result.tile_id ? 4 : 2,
      fillOpacity: selected?.tile_id === result.tile_id ? 0.3 : 0.15
    }} eventHandlers={{ click: () => onSelect(result.tile_id) }}><Tooltip>#{result.rank} {result.category}</Tooltip></Rectangle>)}
    {changePolygons && <GeoJSON data={changePolygons} style={(feature) => ({ color: changeColors[feature.properties?.class_code] || "#888", fillOpacity: 0.5, weight: 1 })}
      onEachFeature={(feature, layer) => layer.bindPopup(`${feature.properties?.class_name || "Change"}: ${(feature.properties?.area_m2 / 10000).toFixed(2)} hectares`)} />}
  </MapContainer>{changePolygons && <Legend />}</div>;
}
