/**
 * ui.js — panel controls, timeline, cell info panel
 */

/** fold number → period label + badge class */
const FOLD_INFO = {
  1: { label: 'Early', cls: 'early' },
  2: { label: 'Mid',   cls: 'mid'   },
  3: { label: 'Late',  cls: 'late'  },
};

/** Format ISO timestamp → "05-04 12:06" */
export function fmtTime(iso) {
  const d = new Date(iso);
  const pad = n => String(n).padStart(2, '0');
  return `${pad(d.getUTCMonth() + 1)}-${pad(d.getUTCDate())} ` +
         `${pad(d.getUTCHours())}:${pad(d.getUTCMinutes())}`;
}

/** Update timeline header (time + fold badge) and meta row */
export function updateTimelineHeader(pair, index, total) {
  const timeEl  = document.getElementById('pair-time');
  const badgeEl = document.getElementById('fold-badge');
  const idxEl   = document.getElementById('pair-index');
  const dtEl    = document.getElementById('pair-delta');

  if (!pair) {
    timeEl.textContent  = '—';
    badgeEl.textContent = '—';
    badgeEl.className   = 'badge';
    idxEl.textContent   = '0 / 0';
    dtEl.textContent    = '—';
    return;
  }

  timeEl.textContent = `${fmtTime(pair.T1)} → ${fmtTime(pair.T2)}`;

  const info = FOLD_INFO[pair.fold] || { label: `F${pair.fold}`, cls: '' };
  badgeEl.textContent = info.label;
  badgeEl.className   = `badge ${info.cls}`;

  idxEl.textContent = `${index + 1} / ${total}`;
  dtEl.textContent  = `Δt = ${pair.delta_t_h.toFixed(1)} h`;
}

/** Set play-button icon ("▶" or "⏸") */
export function setPlayButtonState(playing) {
  const btn = document.getElementById('btn-play');
  btn.textContent = playing ? '⏸' : '▶';
  btn.title = playing ? 'Pause' : 'Play';
}

/** Show pair stats (burned / unburned / cloud counts) */
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
  `;
  el.hidden = false;
}

/** Show hovered cell properties */
export function showCellInfo(info, model = 'xgb') {
  const el      = document.getElementById('cell-info');
  const content = document.getElementById('cell-content');

  if (!info || !info.object) {
    el.hidden = true;
    return;
  }

  const props = info.object.properties;
  const labelText  = { 0: 'Unburned', 1: 'Burned', 2: 'Cloud' }[props.label] ?? '—';
  const labelClass = { 0: 'unburned', 1: 'burned', 2: 'cloud' }[props.label] ?? '';
  const prob       = props[`prob_${model}`];

  const rows = [
    ['Label',             `<span class="badge ${labelClass}">${labelText}</span>`],
    [`Prob (${model.toUpperCase()})`, prob != null ? fmt(prob, '', 3) : '—'],
    ['Dist front',        fmt(props.dist_to_fire_front, 'm', 0)],
    ['Wind align (mean)', fmt(props.wind_alignment_mean, '', 3)],
    ['Wind align (max)',  fmt(props.wind_alignment_max, '', 3)],
    ['Wind spd (path)',   fmt(props.wind_speed_mean, 'm/s', 1)],
    ['Grade',             fmt(props.grade, '', 4)],
    ['Slope (mean)',      fmt(props.slope_mean, '°', 1)],
    ['Slope',             fmt(props.slope, '°', 1)],
    ['Fuel type',         props.fuel_type ?? '—'],
    ['ROS',               fmt(props.ros, 'm/min', 2)],
    ['FFMC',              fmt(props.ffmc, '', 1)],
    ['ISI',               fmt(props.isi, '', 1)],
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
