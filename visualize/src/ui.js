/**
 * ui.js — panel controls, pair/fold selectors, cell info panel
 */

/** Populate fold <select> from pairs index */
export function buildFoldOptions(pairs, foldSelect) {
  const folds = [...new Set(pairs.map(p => p.fold).filter(f => f != null))].sort();
  foldSelect.innerHTML = folds.map(f => `<option value="${f}">Fold ${f}</option>`).join('');
}

/** Populate pair <select> for a given fold */
export function buildPairOptions(pairs, fold, pairSelect) {
  const filtered = pairs.filter(p => p.fold === fold);
  pairSelect.innerHTML = filtered.map(p => {
    const t1 = fmtTime(p.T1);
    const t2 = fmtTime(p.T2);
    const nh = p.n_burned + p.n_unburned + p.n_cloud;
    return `<option value="${p.pair_id}">${t1} → ${t2}  (${nh} cells)</option>`;
  }).join('');
}

/** Format ISO timestamp to readable short form */
function fmtTime(iso) {
  const d = new Date(iso);
  const pad = n => String(n).padStart(2, '0');
  return `${pad(d.getUTCMonth()+1)}-${pad(d.getUTCDate())} ${pad(d.getUTCHours())}:${pad(d.getUTCMinutes())}`;
}

/** Show pair stats (burned/unburned/cloud counts) */
export function showPairStats(pairMeta) {
  const el = document.getElementById('pair-stats');
  const content = document.getElementById('stats-content');
  if (!pairMeta) { el.hidden = true; return; }

  const total = pairMeta.n_burned + pairMeta.n_unburned + pairMeta.n_cloud;
  const pct = n => total > 0 ? ((n / total) * 100).toFixed(1) : '0.0';

  content.innerHTML = `
    <div class="stat-row"><span class="stat-dot burned"></span>
      Burned: <b>${pairMeta.n_burned}</b> <span class="muted">(${pct(pairMeta.n_burned)}%)</span>
    </div>
    <div class="stat-row"><span class="stat-dot unburned"></span>
      Unburned: <b>${pairMeta.n_unburned}</b>
    </div>
    <div class="stat-row"><span class="stat-dot cloud"></span>
      Cloud: <b>${pairMeta.n_cloud}</b>
    </div>
    <div class="stat-row muted">Δt = ${pairMeta.delta_t_h.toFixed(1)} h</div>
  `;
  el.hidden = false;
}

/** Show hovered cell properties */
export function showCellInfo(info) {
  const el      = document.getElementById('cell-info');
  const content = document.getElementById('cell-content');

  if (!info || !info.object) {
    el.hidden = true;
    return;
  }

  const props = info.object.properties;
  const labelText = { 0: 'Unburned', 1: 'Burned', 2: 'Cloud' }[props.label] ?? '—';
  const labelClass = { 0: 'unburned', 1: 'burned', 2: 'cloud' }[props.label] ?? '';

  const rows = [
    ['Label',             `<span class="badge ${labelClass}">${labelText}</span>`],
    // distance
    ['Dist front',        fmt(props.dist_to_fire_front, 'm', 0)],
    // path (A→B)
    ['Wind align (mean)', fmt(props.wind_alignment_mean, '', 3)],
    ['Wind align (max)',  fmt(props.wind_alignment_max, '', 3)],
    ['Wind spd (path)',   fmt(props.wind_speed_mean, 'm/s', 1)],
    ['Grade',             fmt(props.grade, '', 4)],
    ['Slope (mean)',      fmt(props.slope_mean, '°', 1)],
    // static
    ['Slope',             fmt(props.slope, '°', 1)],
    ['Fuel type',         props.fuel_type ?? '—'],
    // FWI
    ['ROS',               fmt(props.ros, 'm/min', 2)],
    ['FFMC',              fmt(props.ffmc, '', 1)],
    ['ISI',               fmt(props.isi, '', 1)],
    // weather
    ['Wind speed',        fmt(props.wind_speed, 'm/s', 1)],
    ['Temp',              fmt(props.temp_c, '°C', 1)],
    ['RH',                fmt(props.rh, '%', 0)],
  ];

  content.innerHTML = rows.map(([k, v]) =>
    `<div class="info-row"><span class="info-key">${k}</span><span class="info-val">${v}</span></div>`
  ).join('');
  el.hidden = false;
}

function fmt(v, unit = '', dec = 2) {
  if (v == null || v !== v) return '—';
  return `${Number(v).toFixed(dec)}${unit ? ' ' + unit : ''}`;
}
