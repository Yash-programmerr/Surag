async function request(url, options = {}) {
  let response;
  try {
    response = await fetch(url, options);
  } catch {
    throw new Error("Network error");
  }
  let body = {};
  try { body = await response.json(); } catch { /* non-JSON error */ }
  if (!response.ok) throw new Error(body.detail || "Request failed");
  return body;
}

export const getHealth = () => request("/api/health");
export const getStats = () => request("/api/stats");
export const reloadData = () => request("/api/reload", { method: "POST" });
export const searchTiles = (payload) => request("/api/search", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(payload)
});
export const getTile = (tileId) => request(`/api/tiles/${encodeURIComponent(tileId)}`);
export const getChange = (tileId) => request(`/api/tiles/${encodeURIComponent(tileId)}/change`);
