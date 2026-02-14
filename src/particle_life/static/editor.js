const canvas = document.getElementById('world');
const ctx = canvas.getContext('2d');
const statusNode = document.getElementById('status');
const typeSelect = document.getElementById('type');
const nameInput = document.getElementById('name');

let cfg = null;
let matrix = null;
let particles = [];
let drawing = false;

function resize() {
  const dpr = window.devicePixelRatio || 1;
  const w = Math.max(1, Math.floor(canvas.clientWidth * dpr));
  const h = Math.max(1, Math.floor(canvas.clientHeight * dpr));
  if (canvas.width !== w || canvas.height !== h) {
    canvas.width = w;
    canvas.height = h;
  }
  draw();
}
window.addEventListener('resize', resize);

function colors() {
  return Array.isArray(cfg?.type_colors) && cfg.type_colors.length ? cfg.type_colors : ['#ff6f5f', '#56c3ff', '#7aff63', '#ffe26a'];
}

function toWorld(event) {
  const rect = canvas.getBoundingClientRect();
  const x = (event.clientX - rect.left) / Math.max(1, rect.width);
  const y = (event.clientY - rect.top) / Math.max(1, rect.height);
  return [x * cfg.world_size, y * cfg.world_size];
}

function addStroke(cx, cy) {
  const radius = Math.max(0.001, Number(document.getElementById('radius').value));
  const density = Math.max(1, Number(document.getElementById('density').value) | 0);
  const jitter = Math.max(0, Number(document.getElementById('jitter').value));
  const speed = Math.max(0, Number(document.getElementById('speed').value));
  const type = Number(typeSelect.value) | 0;
  for (let i = 0; i < density; i += 1) {
    const a = Math.random() * Math.PI * 2;
    const r = radius * Math.sqrt(Math.random());
    const j = 1 + (Math.random() * 2 - 1) * jitter;
    const px = Math.min(cfg.world_size, Math.max(0, cx + Math.cos(a) * r * j));
    const py = Math.min(cfg.world_size, Math.max(0, cy + Math.sin(a) * r * j));
    const va = Math.random() * Math.PI * 2;
    const vm = Math.random() * speed;
    particles.push({ position: [px, py], velocity: [Math.cos(va) * vm, Math.sin(va) * vm], type });
  }
  draw();
}

function draw() {
  if (!cfg) return;
  ctx.fillStyle = '#050913';
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  const scaleX = canvas.width / cfg.world_size;
  const scaleY = canvas.height / cfg.world_size;
  const palette = colors();
  for (const p of particles) {
    ctx.fillStyle = palette[p.type % palette.length];
    ctx.beginPath();
    ctx.arc(p.position[0] * scaleX, p.position[1] * scaleY, 2, 0, Math.PI * 2);
    ctx.fill();
  }
  statusNode.textContent = `particles: ${particles.length}\nspecies: ${cfg.species_count}`;
}

function buildInputJson() {
  const counts = new Array(cfg.species_count).fill(0);
  for (const p of particles) counts[p.type] += 1;
  return {
    schema_version: 1,
    config: { ...cfg, particle_counts: counts },
    interaction_matrix: matrix,
    particles,
  };
}

async function save() {
  const payload = { name: nameInput.value.trim(), input_json: buildInputJson() };
  const res = await fetch('/api/initial_condition/save', {
    method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify(payload),
  });
  const body = await res.json();
  statusNode.textContent = body.ok ? `saved: ${body.name}\nparticles: ${particles.length}` : `error: ${body.error}`;
}

function download() {
  const blob = new Blob([JSON.stringify(buildInputJson())], { type: 'application/json' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = nameInput.value.trim() || 'initial_condition.json';
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 250);
}

async function init() {
  const r = await fetch('/api/config');
  const data = await r.json();
  cfg = data.values;
  matrix = data.values.interaction_matrix;
  for (let i = 0; i < cfg.species_count; i += 1) {
    const o = document.createElement('option');
    o.value = String(i);
    o.textContent = `Type ${i + 1}`;
    typeSelect.appendChild(o);
  }
  resize();
}

canvas.addEventListener('mousedown', (e) => { drawing = true; const [x,y] = toWorld(e); addStroke(x,y); });
canvas.addEventListener('mousemove', (e) => { if (!drawing) return; const [x,y] = toWorld(e); addStroke(x,y); });
window.addEventListener('mouseup', () => { drawing = false; });

document.getElementById('clear').addEventListener('click', () => { particles = []; draw(); });
document.getElementById('save').addEventListener('click', save);
document.getElementById('download').addEventListener('click', download);

init();
