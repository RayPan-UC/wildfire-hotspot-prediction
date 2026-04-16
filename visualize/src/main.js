/**
 * main.js — map init + layer orchestration + timeline playback
 */
import maplibregl from 'maplibre-gl';
import 'maplibre-gl/dist/maplibre-gl.css';
import './styles/main.scss';
import { MapboxOverlay as DeckOverlay } from '@deck.gl/mapbox';

import { fetchMeta, fetchPairsIndex, fetchBoundary, fetchPairLayer } from './data.js';
import { boundaryLayer, selectorLayer, receptorsLayer, sourcesLayer, burnedMarkersLayer } from './layers.js';
import {
  updateTimelineHeader, setPlayButtonState,
  showPairStats, showCellInfo,
} from './ui.js';

// ── State ──────────────────────────────────────────────────────────────────
let deck;
let allPairs    = [];    // sorted chronologically by T1
let currentIdx  = 0;
let currentPair = null;
let model       = 'xgb';
let thresholds  = {};    // { fold: { xgb: t, rf: t, lr: t } }

// Playback state
let playing   = false;
let playTimer = null;
let stepMs    = 1000;

// Layer visibility flags
const vis = {
  boundary:  true,
  selector:  true,
  receptors: true,
  sources:   true,
  burned:    true,
};

// Loaded GeoJSON cache for current pair
const geo = {
  boundary:  null,
  selector:  null,
  receptors: null,
  sources:   null,
};

// Track in-flight load so rapid scrubbing ignores stale responses
let loadSeq = 0;

// ── Init ───────────────────────────────────────────────────────────────────
async function init() {
  const meta = await fetchMeta();
  document.getElementById('study-name').textContent = meta.study_name;
  thresholds = meta.thresholds || {};

  // MapLibre
  const [lon, lat] = meta.center;
  const map = new maplibregl.Map({
    container: 'map',
    style:     'https://basemaps.cartocdn.com/gl/positron-gl-style/style.json',
    center:    [lon, lat],
    zoom:      8,
  });
  map.addControl(new maplibregl.NavigationControl(), 'bottom-right');

  // deck.gl overlay
  deck = new DeckOverlay({
    interleaved: true,
    layers: [],
    getTooltip: () => null,
  });
  map.addControl(deck);

  // Load pairs index (sorted by T1)
  const raw = await fetchPairsIndex();
  allPairs = raw.slice().sort((a, b) => new Date(a.T1) - new Date(b.T1));

  // Timeline slider
  const slider = document.getElementById('pair-slider');
  slider.min   = '0';
  slider.max   = String(allPairs.length - 1);
  slider.value = '0';
  slider.addEventListener('input', () => {
    pause();
    goTo(Number(slider.value));
  });

  // Playback buttons
  document.getElementById('btn-play').addEventListener('click', togglePlay);
  document.getElementById('btn-prev').addEventListener('click', () => { pause(); goTo(currentIdx - 1); });
  document.getElementById('btn-next').addEventListener('click', () => { pause(); goTo(currentIdx + 1); });
  document.getElementById('speed-select').addEventListener('change', e => {
    stepMs = Number(e.target.value);
    if (playing) restartTimer();
  });

  // Model selector
  document.querySelectorAll('input[name="model"]').forEach(r => {
    r.addEventListener('change', e => {
      model = e.target.value;
      renderLayers();
    });
  });

  // Layer toggles
  ['boundary', 'selector', 'receptors', 'sources', 'burned'].forEach(id => {
    document.getElementById(`layer-${id}`).addEventListener('change', e => {
      vis[id] = e.target.checked;
      renderLayers();
    });
  });

  // Keyboard shortcuts: space = play/pause, ← / → step
  document.addEventListener('keydown', e => {
    if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT') return;
    if (e.code === 'Space')       { e.preventDefault(); togglePlay(); }
    else if (e.code === 'ArrowLeft')  { pause(); goTo(currentIdx - 1); }
    else if (e.code === 'ArrowRight') { pause(); goTo(currentIdx + 1); }
  });

  // Initial render
  goTo(0);
}

// ── Timeline navigation ────────────────────────────────────────────────────
async function goTo(idx) {
  if (allPairs.length === 0) return;
  idx = Math.max(0, Math.min(allPairs.length - 1, idx));
  currentIdx  = idx;
  currentPair = allPairs[idx];

  document.getElementById('pair-slider').value = String(idx);
  updateTimelineHeader(currentPair, idx, allPairs.length);
  showPairStats(currentPair);
  showCellInfo(null);

  const mySeq = ++loadSeq;
  const [boundary, selector, receptors, sources] = await Promise.all([
    fetchBoundary(currentPair.T1),
    fetchPairLayer(currentPair.pair_id, 'selector'),
    fetchPairLayer(currentPair.pair_id, 'receptors'),
    fetchPairLayer(currentPair.pair_id, 'sources'),
  ]);
  if (mySeq !== loadSeq) return;   // superseded by a newer navigation

  geo.boundary  = boundary;
  geo.selector  = selector;
  geo.receptors = receptors;
  geo.sources   = sources;

  renderLayers();
}

// ── Playback control ───────────────────────────────────────────────────────
function togglePlay() {
  if (playing) pause();
  else         play();
}

function play() {
  if (playing) return;
  playing = true;
  setPlayButtonState(true);
  // Wrap to start if playing from end
  if (currentIdx >= allPairs.length - 1) goTo(0);
  restartTimer();
}

function pause() {
  playing = false;
  setPlayButtonState(false);
  if (playTimer) { clearInterval(playTimer); playTimer = null; }
}

function restartTimer() {
  if (playTimer) clearInterval(playTimer);
  playTimer = setInterval(() => {
    if (currentIdx >= allPairs.length - 1) { pause(); return; }
    goTo(currentIdx + 1);
  }, stepMs);
}

// ── Render deck.gl layers ──────────────────────────────────────────────────
function renderLayers() {
  const fold = currentPair ? currentPair.fold : null;
  const thr  = (fold != null && thresholds[fold] && thresholds[fold][model] != null)
    ? thresholds[fold][model]
    : null;
  deck.setProps({
    layers: [
      boundaryLayer(geo.boundary,   vis.boundary),
      selectorLayer(geo.selector,   vis.selector),
      receptorsLayer(geo.receptors, vis.receptors, model, info => showCellInfo(info, model), thr),
      burnedMarkersLayer(geo.receptors, vis.burned),
      sourcesLayer(geo.sources,     vis.sources),
    ].filter(Boolean),
  });
}

init();
