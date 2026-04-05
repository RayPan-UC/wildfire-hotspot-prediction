/**
 * data.js — fetch helpers for data_render/ API
 * All paths are served under /data/ by the Python server.
 */

export async function fetchMeta() {
  const r = await fetch('/data/meta.json');
  return r.json();
}

export async function fetchPairsIndex() {
  const r = await fetch('/data/pairs/index.json');
  return r.json();
}

export async function fetchFireGrowth() {
  const r = await fetch('/data/fire_growth.json');
  return r.json();
}

export async function fetchBoundary(t) {
  // t is an ISO string like "2016-05-04T0612"
  const key = t.replace(/[-:]/g, '').slice(0, 13).replace('T', 'T');
  // Normalise to filename format YYYY-MM-DDTHHMM
  const dt  = new Date(t);
  const pad = n => String(n).padStart(2, '0');
  const fname = `${dt.getUTCFullYear()}-${pad(dt.getUTCMonth()+1)}-${pad(dt.getUTCDate())}` +
                `T${pad(dt.getUTCHours())}${pad(dt.getUTCMinutes())}`;
  const r = await fetch(`/data/boundaries/${fname}.geojson`);
  if (!r.ok) return null;
  return r.json();
}

export async function fetchPairLayer(pairId, layer) {
  // layer: 'receptors' | 'sources' | 'selector'
  const r = await fetch(`/data/pairs/${pairId}/${layer}.geojson`);
  if (!r.ok) return null;
  return r.json();
}

export async function fetchPredictions(pairId) {
  const r = await fetch(`/data/predictions/${pairId}/predicted.geojson`);
  if (!r.ok) return null;
  return r.json();
}
