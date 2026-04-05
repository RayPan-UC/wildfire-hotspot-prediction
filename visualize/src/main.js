/**
 * main.js — map init + layer orchestration
 *
 * Stack:
 *   MapLibre GL  →  base map (Carto Positron, light)
 *   deck.gl MapboxOverlay  →  data layers (interleaved)
 */
import maplibregl from 'maplibre-gl';
import 'maplibre-gl/dist/maplibre-gl.css';
import './styles/main.scss';
import { MapboxOverlay as DeckOverlay } from '@deck.gl/mapbox';

import { fetchMeta, fetchPairsIndex, fetchBoundary, fetchPairLayer } from './data.js';
import { boundaryLayer, selectorLayer, receptorsLayer, sourcesLayer } from './layers.js';
import { buildFoldOptions, buildPairOptions, showPairStats, showCellInfo } from './ui.js';

// ── State ──────────────────────────────────────────────────────────────────
let deck;
let allPairs    = [];
let currentPair = null;

// Layer visibility flags
const vis = {
  boundary:  true,
  selector:  true,
  receptors: true,
  sources:   true,
};

// Loaded GeoJSON cache for current pair
const geo = {
  boundary: null,
  selector: null,
  receptors: null,
  sources:   null,
};

// ── Init ───────────────────────────────────────────────────────────────────
async function init() {
  const meta = await fetchMeta();

  // Update page title
  document.getElementById('study-name').textContent = meta.study_name;

  // Init MapLibre
  const [lon, lat] = meta.center;
  const map = new maplibregl.Map({
    container: 'map',
    style:     'https://basemaps.cartocdn.com/gl/positron-gl-style/style.json',
    center:    [lon, lat],
    zoom:      8,
  });

  map.addControl(new maplibregl.NavigationControl(), 'bottom-right');

  // Init deck.gl overlay (interleaved with MapLibre)
  deck = new DeckOverlay({
    interleaved: true,
    layers: [],
    getTooltip: () => null,   // we use our own panel
  });
  map.addControl(deck);

  // Load pairs index
  allPairs = await fetchPairsIndex();

  // Populate fold / pair selectors
  const foldSel = document.getElementById('fold-select');
  const pairSel = document.getElementById('pair-select');

  buildFoldOptions(allPairs, foldSel);
  buildPairOptions(allPairs, Number(foldSel.value), pairSel);

  foldSel.addEventListener('change', () => {
    buildPairOptions(allPairs, Number(foldSel.value), pairSel);
    loadPair(pairSel.value);
  });
  pairSel.addEventListener('change', () => loadPair(pairSel.value));

  // Layer toggles
  ['boundary', 'selector', 'receptors', 'sources'].forEach(id => {
    document.getElementById(`layer-${id}`).addEventListener('change', e => {
      vis[id] = e.target.checked;
      renderLayers();
    });
  });

  // Load first pair
  if (pairSel.value) loadPair(pairSel.value);
}

// ── Load a pair ────────────────────────────────────────────────────────────
async function loadPair(pairId) {
  currentPair = allPairs.find(p => String(p.pair_id) === String(pairId));
  if (!currentPair) return;

  showPairStats(currentPair);
  showCellInfo(null);

  // Fetch all layers in parallel
  const [boundary, selector, receptors, sources] = await Promise.all([
    fetchBoundary(currentPair.T1),
    fetchPairLayer(pairId, 'selector'),
    fetchPairLayer(pairId, 'receptors'),
    fetchPairLayer(pairId, 'sources'),
  ]);

  geo.boundary  = boundary;
  geo.selector  = selector;
  geo.receptors = receptors;
  geo.sources   = sources;

  renderLayers();
}

// ── Render deck.gl layers ──────────────────────────────────────────────────
function renderLayers() {
  deck.setProps({
    layers: [
      boundaryLayer(geo.boundary,  vis.boundary),
      selectorLayer(geo.selector,  vis.selector),
      receptorsLayer(geo.receptors, vis.receptors, info => showCellInfo(info)),
      sourcesLayer(geo.sources,    vis.sources),
    ].filter(Boolean),
  });
}

init();
