/**
 * layers.js — deck.gl layer factories
 */
import { GeoJsonLayer, ScatterplotLayer, TextLayer } from '@deck.gl/layers';

const LABEL_COLOR = {
  0: [59, 130, 246, 200],   // blue  — unburned
  1: [239, 68, 68, 220],    // red   — burned
  2: [156, 163, 175, 160],  // gray  — cloud
};

/**
 * Fire boundary fill + outline.
 */
export function boundaryLayer(data, visible) {
  if (!data) return null;
  return new GeoJsonLayer({
    id: 'boundary',
    data,
    visible,
    filled:      true,
    stroked:     true,
    getFillColor:   [249, 115, 22, 60],   // orange, semi-transparent
    getLineColor:   [249, 115, 22, 200],
    getLineWidth:   80,
    lineWidthUnits: 'meters',
    pickable: false,
  });
}

/**
 * Receptor selector polygon outline.
 */
export function selectorLayer(data, visible) {
  if (!data) return null;
  return new GeoJsonLayer({
    id: 'selector',
    data,
    visible,
    filled:      true,
    stroked:     true,
    getFillColor:   [250, 204, 21, 25],   // yellow, very transparent
    getLineColor:   [250, 204, 21, 180],
    getLineWidth:   60,
    lineWidthUnits: 'meters',
    pickable: false,
  });
}

/**
 * Receptor B cells — colored by predicted probability for the selected model
 * (prob_<model>) if present, else falls back to label color.
 */
export function receptorsLayer(data, visible, model, onHover, threshold = null) {
  if (!data) return null;
  const probKey = `prob_${model}`;
  let src = data;
  if (threshold != null && data.features) {
    src = {
      ...data,
      features: data.features.filter(f => {
        const p = f.properties[probKey];
        return p != null && !Number.isNaN(p) && p >= threshold;
      }),
    };
  }
  return new GeoJsonLayer({
    id: 'receptors',
    data: src,
    visible,
    pointType:        'circle',
    getPointRadius:   220,
    pointRadiusUnits: 'meters',
    filled:           true,
    stroked:          false,
    getFillColor:     f => {
      const p = f.properties[probKey];
      if (p != null && !Number.isNaN(p)) return probToColor(p);
      return LABEL_COLOR[f.properties.label] ?? LABEL_COLOR[0];
    },
    pickable:         true,
    autoHighlight:    true,
    highlightColor:   [255, 255, 255, 180],
    onHover,
    updateTriggers:   { getFillColor: [model], getPointRadius: [threshold] },
  });
}

/** Predicted probability 0..1 → red heat ramp (yellow → orange → red). */
function probToColor(p) {
  p = Math.max(0, Math.min(1, p));
  // linear interp between yellow (250,204,21) and red (239,68,68)
  const r = Math.round(250 + (239 - 250) * p);
  const g = Math.round(204 + ( 68 - 204) * p);
  const b = Math.round( 21 + ( 68 -  21) * p);
  const a = Math.round(120 + 120 * p);   // transparent for low prob, opaque for high
  return [r, g, b, a];
}

/**
 * Burned markers — 🔥 emoji on every receptor with label === 1 (actual T2 fire).
 * Uses the same receptors GeoJSON but filters to burned cells only.
 */
export function burnedMarkersLayer(data, visible) {
  if (!data || !data.features) return null;
  const burned = data.features.filter(f => f.properties.label === 1);
  if (burned.length === 0) return null;
  return new ScatterplotLayer({
    id: 'burned-markers',
    data: burned,
    visible,
    getPosition:      f => f.geometry.coordinates,
    getRadius:        8,
    radiusUnits:      'pixels',
    radiusMinPixels:  5,
    radiusMaxPixels:  14,
    filled:           true,
    stroked:          true,
    getFillColor:     [255, 80, 0, 255],
    getLineColor:     [255, 255, 255, 230],
    getLineWidth:     2,
    lineWidthUnits:   'pixels',
    pickable:         false,
    parameters:       { depthTest: false },
  });
}

/**
 * Source A hotspot points — orange stars (rendered as circles).
 */
export function sourcesLayer(data, visible) {
  if (!data) return null;
  return new GeoJsonLayer({
    id: 'sources',
    data,
    visible,
    pointType:        'circle',
    getPointRadius:   350,
    pointRadiusUnits: 'meters',
    filled:           true,
    stroked:          true,
    getFillColor:     [249, 115, 22, 240],
    getLineColor:     [255, 255, 255, 200],
    getLineWidth:     40,
    lineWidthUnits:   'meters',
    pickable:         false,
  });
}
