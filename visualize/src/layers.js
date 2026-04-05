/**
 * layers.js — deck.gl layer factories
 */
import { GeoJsonLayer, ScatterplotLayer } from '@deck.gl/layers';

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
 * Receptor B cells — colored by label, hover shows cell info.
 */
export function receptorsLayer(data, visible, onHover) {
  if (!data) return null;
  return new GeoJsonLayer({
    id: 'receptors',
    data,
    visible,
    pointType:      'circle',
    getPointRadius: 220,
    pointRadiusUnits: 'meters',
    filled:         true,
    stroked:        false,
    getFillColor:   f => LABEL_COLOR[f.properties.label] ?? LABEL_COLOR[0],
    pickable:       true,
    autoHighlight:  true,
    highlightColor: [255, 255, 255, 180],
    onHover,
    updateTriggers: { getFillColor: [] },
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
