/**
 * growth.js — fire growth time series chart (fire_growth.html)
 * Uses Canvas 2D API directly — no Chart.js dependency needed.
 */
import { fetchMeta, fetchFireGrowth } from './data.js';

async function init() {
  const [meta, rows] = await Promise.all([fetchMeta(), fetchFireGrowth()]);
  document.getElementById('study-name').textContent = meta.study_name;

  const canvas = document.getElementById('growth-chart');
  drawChart(canvas, rows);
}

function drawChart(canvas, rows) {
  const dpr    = window.devicePixelRatio || 1;
  const W      = canvas.parentElement.clientWidth;
  const H      = canvas.parentElement.clientHeight;
  canvas.width  = W * dpr;
  canvas.height = H * dpr;
  canvas.style.width  = W + 'px';
  canvas.style.height = H + 'px';

  const ctx = canvas.getContext('2d');
  ctx.scale(dpr, dpr);

  const pad   = { top: 40, right: 30, bottom: 60, left: 70 };
  const w     = W - pad.left - pad.right;
  const h     = H - pad.top  - pad.bottom;

  const times = rows.map(r => new Date(r.time).getTime());
  const areas = rows.map(r => r.area_km2);

  const xMin = Math.min(...times), xMax = Math.max(...times);
  const yMax = Math.max(...areas) * 1.05;

  const xScale = t  => pad.left + (t  - xMin) / (xMax - xMin) * w;
  const yScale = v  => pad.top  + h - v / yMax * h;

  // Background
  ctx.fillStyle = '#f8fafc';
  ctx.fillRect(0, 0, W, H);

  // Grid lines + y labels
  ctx.strokeStyle = '#e2e8f0';
  ctx.fillStyle   = '#64748b';
  ctx.font        = '11px system-ui, sans-serif';
  ctx.textAlign   = 'right';

  const yTicks = 5;
  for (let i = 0; i <= yTicks; i++) {
    const v = (yMax / yTicks) * i;
    const y = yScale(v);
    ctx.beginPath();
    ctx.moveTo(pad.left, y);
    ctx.lineTo(pad.left + w, y);
    ctx.stroke();
    ctx.fillText(v.toFixed(0), pad.left - 8, y + 4);
  }

  // Y axis label
  ctx.save();
  ctx.translate(16, pad.top + h / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.textAlign = 'center';
  ctx.fillStyle = '#475569';
  ctx.font = '12px system-ui, sans-serif';
  ctx.fillText('Burned area (km²)', 0, 0);
  ctx.restore();

  // X labels (sample every few rows)
  ctx.textAlign = 'center';
  ctx.fillStyle = '#64748b';
  ctx.font = '10px system-ui, sans-serif';
  const xStep = Math.max(1, Math.floor(rows.length / 8));
  rows.forEach((r, i) => {
    if (i % xStep !== 0) return;
    const d = new Date(r.time);
    const label = `${d.getUTCMonth()+1}/${d.getUTCDate()} ${String(d.getUTCHours()).padStart(2,'0')}h`;
    ctx.fillText(label, xScale(times[i]), pad.top + h + 20);
  });

  // Area fill
  ctx.beginPath();
  ctx.moveTo(xScale(times[0]), yScale(0));
  rows.forEach((r, i) => ctx.lineTo(xScale(times[i]), yScale(areas[i])));
  ctx.lineTo(xScale(times[times.length - 1]), yScale(0));
  ctx.closePath();
  ctx.fillStyle = 'rgba(249,115,22,0.15)';
  ctx.fill();

  // Line
  ctx.beginPath();
  rows.forEach((r, i) => {
    const x = xScale(times[i]), y = yScale(areas[i]);
    i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
  });
  ctx.strokeStyle = '#f97316';
  ctx.lineWidth   = 2.5;
  ctx.lineJoin    = 'round';
  ctx.stroke();

  // Dots
  rows.forEach((r, i) => {
    ctx.beginPath();
    ctx.arc(xScale(times[i]), yScale(areas[i]), 3, 0, Math.PI * 2);
    ctx.fillStyle = '#f97316';
    ctx.fill();
  });

  // Title
  ctx.textAlign = 'center';
  ctx.fillStyle = '#1e293b';
  ctx.font = 'bold 14px system-ui, sans-serif';
  ctx.fillText('Fire Growth Over Time', pad.left + w / 2, 22);
}

init();
