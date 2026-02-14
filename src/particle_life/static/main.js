const canvas = document.getElementById("canvas");
const panel = document.getElementById("panel");
const gl = canvas.getContext("webgl");
if (!gl) throw new Error("WebGL unavailable");

const TYPE_COLORS = ["#ff6f5f", "#56c3ff", "#7aff63", "#ffe26a", "#d086ff", "#ffa04d", "#60ffd0", "#c8dbff", "#ff73b8", "#88ff9e", "#6f8bff", "#ffc56f"];
const DEV_MODE = new URLSearchParams(window.location.search).has("dev");

/** @typedef {{sections: Array<object>, presets: string[], values: Record<string, any>}} SimulationParams */

const appState = {
  sections: [],
  presets: [],
  values: {},
  matrixDraft: null,
  sectionCollapsed: new Map(),
  paused: false,
  subscribers: new Set(),
};
let pointCount = 0;
let positions = new Float32Array();
let species = new Float32Array();
let velocities = new Float32Array();
let matrixGrid;
let statsNodes = {};
let latestUpdateToken = 0;
let latestAppliedUpdateToken = 0;
let matrixUpdateInFlight = false;
let matrixUpdateQueued = false;

const perf = { gfxFrames: 0, physicsFrames: 0, lastStamp: performance.now(), gfxFps: 0, physicsFps: 0 };

const vertexSrc = `
attribute vec2 a_pos;
attribute float a_species;
attribute vec2 a_vel;
uniform vec2 u_scale;
uniform float u_pointSize;
uniform int u_colorMode;
varying vec4 v_color;
vec3 speciesColor(float i) {
  int idx = int(mod(i, 8.0));
  if (idx == 0) return vec3(1.0, 0.45, 0.35);
  if (idx == 1) return vec3(0.4, 0.85, 1.0);
  if (idx == 2) return vec3(0.62, 1.0, 0.45);
  if (idx == 3) return vec3(1.0, 0.95, 0.45);
  if (idx == 4) return vec3(0.88, 0.52, 1.0);
  if (idx == 5) return vec3(1.0, 0.7, 0.25);
  if (idx == 6) return vec3(0.6, 1.0, 0.9);
  return vec3(0.8, 0.85, 1.0);
}
void main() {
  vec2 clip = a_pos * 2.0 - 1.0;
  gl_Position = vec4(clip.x, -clip.y, 0.0, 1.0);
  gl_PointSize = u_pointSize * u_scale.x;
  if (u_colorMode == 1) {
    float s = clamp(length(a_vel) * 30.0, 0.0, 1.0);
    v_color = vec4(vec3(s, 0.2, 1.0 - s), 1.0);
  } else if (u_colorMode == 2) {
    v_color = vec4(vec3(0.93), 1.0);
  } else {
    v_color = vec4(speciesColor(a_species), 1.0);
  }
}`;

const fragmentSrc = `
precision mediump float;
varying vec4 v_color;
uniform float u_opacity;
void main() {
  vec2 c = gl_PointCoord - vec2(0.5);
  if (dot(c, c) > 0.25) discard;
  gl_FragColor = vec4(v_color.rgb, u_opacity);
}`;

function compile(type, src) {
  const shader = gl.createShader(type);
  gl.shaderSource(shader, src);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(shader));
  return shader;
}

const program = gl.createProgram();
gl.attachShader(program, compile(gl.VERTEX_SHADER, vertexSrc));
gl.attachShader(program, compile(gl.FRAGMENT_SHADER, fragmentSrc));
gl.linkProgram(program);
if (!gl.getProgramParameter(program, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(program));

const posBuffer = gl.createBuffer();
const speciesBuffer = gl.createBuffer();
const velBuffer = gl.createBuffer();
const aPos = gl.getAttribLocation(program, "a_pos");
const aSpecies = gl.getAttribLocation(program, "a_species");
const aVel = gl.getAttribLocation(program, "a_vel");
const uScale = gl.getUniformLocation(program, "u_scale");
const uPointSize = gl.getUniformLocation(program, "u_pointSize");
const uOpacity = gl.getUniformLocation(program, "u_opacity");
const uColorMode = gl.getUniformLocation(program, "u_colorMode");

function resize() {
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.floor(window.innerWidth * dpr);
  canvas.height = Math.floor(window.innerHeight * dpr);
  gl.viewport(0, 0, canvas.width, canvas.height);
}
window.addEventListener("resize", resize);
resize();

function notify() { appState.subscribers.forEach((fn) => fn(appState)); }
function setRemoteState(next) {
  appState.sections = next.sections || appState.sections;
  appState.presets = next.presets || appState.presets;
  appState.values = next.values;
  appState.matrixDraft = next.values.interaction_matrix.map((r) => r.map((v) => Number(v)));
  validateParams();
  notify();
}

function valueText(v) {
  const n = Number(v);
  return Number.isInteger(n) ? String(n) : n.toFixed(3).replace(/0+$/, "").replace(/\.$/, "");
}

function clampMatrixValue(v) {
  const n = Number(v);
  if (!Number.isFinite(n)) return 0;
  return Math.max(-1, Math.min(1, n));
}

function matrixColor(v) {
  const val = clampMatrixValue(v);
  const intensity = Math.floor(Math.abs(val) * 180) + 35;
  return val >= 0 ? `rgb(${30}, ${intensity}, ${35})` : `rgb(${intensity}, ${32}, ${32})`;
}

async function postJSON(path, body = {}) {
  const r = await fetch(path, { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(body) });
  return r.json();
}

async function applyUpdates(updates) {
  const updateToken = ++latestUpdateToken;
  const result = await postJSON("/api/config/update", { updates });
  if (updateToken < latestAppliedUpdateToken) return;
  latestAppliedUpdateToken = updateToken;
  setRemoteState({ values: result.values });
}

async function setPaused(paused) {
  const result = await postJSON("/api/sim/pause", { paused });
  appState.paused = Boolean(result.paused);
  buildUI();
}

function pushMatrixUpdate() {
  if (matrixUpdateInFlight) {
    matrixUpdateQueued = true;
    return;
  }
  matrixUpdateInFlight = true;
  applyUpdates({ interaction_matrix: appState.matrixDraft.map((row) => [...row]) })
    .finally(() => {
      matrixUpdateInFlight = false;
      if (matrixUpdateQueued) {
        matrixUpdateQueued = false;
        pushMatrixUpdate();
      }
    });
}

async function applyControlValue(key, raw, control) {
  let value = raw;
  if (control.type === "range" || control.type === "number") value = Number(raw);
  if (control.type === "toggle") value = Boolean(raw);
  await applyUpdates({ [key]: value });
}

function createSection(title, bodyBuilder) {
  const section = document.createElement("div");
  section.className = "section";
  const header = document.createElement("div");
  header.className = "section-header";
  const key = title.toLowerCase();
  const chevron = document.createElement("span");
  chevron.className = "collapse-chevron";
  chevron.textContent = "▾";
  header.append(title, chevron);

  const body = document.createElement("div");
  body.className = "section-body";
  bodyBuilder(body);

  const collapsed = appState.sectionCollapsed.get(key) ?? false;
  if (collapsed) {
    header.dataset.collapsed = "true";
    body.style.display = "none";
    chevron.textContent = "▸";
  }
  header.addEventListener("click", () => {
    const nowCollapsed = body.style.display !== "none";
    appState.sectionCollapsed.set(key, nowCollapsed);
    body.style.display = nowCollapsed ? "none" : "block";
    header.dataset.collapsed = nowCollapsed ? "true" : "false";
    chevron.textContent = nowCollapsed ? "▸" : "▾";
  });
  section.append(header, body);
  panel.appendChild(section);
}

function bindControl(parent, control) {
  const row = document.createElement("div");
  row.className = "field-row";
  const label = document.createElement("label");
  label.textContent = control.label;
  row.appendChild(label);

  const valueNode = document.createElement("span");
  valueNode.className = "value";
  const applyTag = document.createElement("span");
  applyTag.className = "apply-tag";
  applyTag.textContent = control.apply || "";

  if (control.type === "range") {
    const input = document.createElement("input");
    input.type = "range";
    input.min = control.min;
    input.max = control.max;
    input.step = control.step;
    input.value = appState.values[control.key] ?? control.default;
    valueNode.textContent = valueText(input.value);
    input.addEventListener("input", () => { valueNode.textContent = valueText(input.value); });
    input.addEventListener("change", () => applyControlValue(control.key, input.value, control));
    row.appendChild(input);
  } else if (control.type === "select") {
    const select = document.createElement("select");
    control.options.forEach((opt) => {
      const o = document.createElement("option");
      o.value = opt;
      o.textContent = opt;
      select.appendChild(o);
    });
    select.value = String(appState.values[control.key] ?? control.default);
    valueNode.textContent = select.value;
    select.addEventListener("change", () => {
      valueNode.textContent = select.value;
      applyControlValue(control.key, select.value, control);
    });
    row.appendChild(select);
  } else if (control.type === "toggle") {
    const input = document.createElement("input");
    input.type = "checkbox";
    input.checked = Boolean(appState.values[control.key] ?? control.default);
    valueNode.textContent = input.checked ? "on" : "off";
    input.addEventListener("change", () => {
      valueNode.textContent = input.checked ? "on" : "off";
      applyControlValue(control.key, input.checked, control);
    });
    row.appendChild(input);
  } else if (control.type === "number") {
    const input = document.createElement("input");
    input.type = "number";
    input.min = control.min;
    input.max = control.max;
    input.step = control.step;
    input.value = String(appState.values[control.key] ?? control.default);
    valueNode.textContent = input.value;
    input.addEventListener("change", () => {
      valueNode.textContent = input.value;
      applyControlValue(control.key, input.value, control);
    });
    row.appendChild(input);
  }

  row.append(valueNode, applyTag);
  parent.appendChild(row);
}


function buildParticleCountControls(parent) {
  const speciesCount = Number(appState.values.species_count || 0);
  const defaultCount = Number(appState.values.particles_per_species || 0);
  const counts = Array.isArray(appState.values.particle_counts) ? appState.values.particle_counts.slice(0, speciesCount) : [];
  while (counts.length < speciesCount) counts.push(defaultCount);

  const hint = document.createElement("div");
  hint.className = "mini";
  hint.textContent = "Per-type particle counts (applies on reset)";
  parent.appendChild(hint);

  counts.forEach((count, idx) => {
    const row = document.createElement("div");
    row.className = "field-row";

    const label = document.createElement("label");
    label.textContent = `Type ${idx + 1}`;

    const input = document.createElement("input");
    input.type = "number";
    input.min = "0";
    input.max = "400";
    input.step = "1";
    input.value = String(count);
    input.style.width = "88px";

    const valueNode = document.createElement("span");
    valueNode.className = "value";
    valueNode.textContent = input.value;

    const applyTag = document.createElement("span");
    applyTag.className = "apply-tag";
    applyTag.textContent = "reset";

    input.addEventListener("change", () => {
      valueNode.textContent = input.value;
      const nextCounts = counts.slice();
      nextCounts[idx] = Number(input.value);
      applyUpdates({ particle_counts: nextCounts });
    });

    row.append(label, input, valueNode, applyTag);
    parent.appendChild(row);
  });
}

function buildMatrixEditor(parent) {
  const toolbar = document.createElement("div");
  toolbar.className = "matrix-toolbar";

  const presetSelect = document.createElement("select");
  [["fully_random", "fully random"], ["zero", "all zeros"], ["identity", "identity +"], ["symmetric_random", "symmetric random"]].forEach(([value, label]) => {
    const o = document.createElement("option");
    o.value = value;
    o.textContent = label;
    presetSelect.appendChild(o);
  });

  const presetApply = document.createElement("button");
  presetApply.textContent = "Apply preset";
  presetApply.addEventListener("click", () => {
    const matrix = appState.matrixDraft.map((r) => [...r]);
    const n = matrix.length;
    if (presetSelect.value === "fully_random") for (let i = 0; i < n; i += 1) for (let j = 0; j < n; j += 1) matrix[i][j] = -1 + Math.random() * 2;
    if (presetSelect.value === "zero") for (let i = 0; i < n; i += 1) for (let j = 0; j < n; j += 1) matrix[i][j] = 0;
    if (presetSelect.value === "identity") for (let i = 0; i < n; i += 1) for (let j = 0; j < n; j += 1) matrix[i][j] = i === j ? 1 : 0;
    if (presetSelect.value === "symmetric_random") {
      for (let i = 0; i < n; i += 1) for (let j = i; j < n; j += 1) { const value = -1 + Math.random() * 2; matrix[i][j] = value; matrix[j][i] = value; }
    }
    appState.matrixDraft = matrix.map((r) => r.map(clampMatrixValue));
    syncMatrix();
    pushMatrixUpdate();
  });

  toolbar.append(presetSelect, presetApply);

  matrixGrid = document.createElement("div");
  matrixGrid.className = "matrix-grid";
  parent.append(toolbar, matrixGrid);
}

function syncMatrix() {
  if (!matrixGrid || !appState.matrixDraft) return;
  const matrix = appState.matrixDraft;
  const n = matrix.length;
  matrixGrid.innerHTML = "";
  matrixGrid.style.gridTemplateColumns = `16px repeat(${n}, minmax(38px, 1fr))`;

  matrixGrid.appendChild(document.createElement("div"));
  for (let c = 0; c < n; c += 1) {
    const dot = document.createElement("div");
    dot.className = "matrix-axis-dot";
    dot.style.background = TYPE_COLORS[c % TYPE_COLORS.length];
    matrixGrid.appendChild(dot);
  }

  for (let i = 0; i < n; i += 1) {
    const rowDot = document.createElement("div");
    rowDot.className = "matrix-axis-dot";
    rowDot.style.background = TYPE_COLORS[i % TYPE_COLORS.length];
    matrixGrid.appendChild(rowDot);

    for (let j = 0; j < n; j += 1) {
      const input = document.createElement("input");
      input.type = "number";
      input.className = "matrix-cell";
      input.step = "0.01";
      input.min = "-1";
      input.max = "1";
      input.value = matrix[i][j].toFixed(3);
      input.style.background = matrixColor(matrix[i][j]);
      input.title = `${i} → ${j}`;
      const applyCellValue = (rawValue, commit = true) => {
        const parsed = Number(rawValue);
        if (!Number.isFinite(parsed)) return;
        const next = clampMatrixValue(parsed);
        if (next === appState.matrixDraft[i][j]) return;
        appState.matrixDraft[i][j] = next;
        input.style.background = matrixColor(next);
        if (commit) pushMatrixUpdate();
      };

      input.addEventListener("input", () => {
        applyCellValue(input.value, true);
      });

      input.addEventListener("change", () => {
        input.value = appState.matrixDraft[i][j].toFixed(3);
      });

      input.addEventListener("mousedown", (event) => {
        if (event.button !== 0 || event.shiftKey) return;
        const startY = event.clientY;
        const startX = event.clientX;
        const startValue = appState.matrixDraft[i][j];
        const onMove = (moveEvent) => {
          const delta = ((startY - moveEvent.clientY) + (moveEvent.clientX - startX)) * 0.004;
          const value = clampMatrixValue(startValue + delta);
          input.value = value.toFixed(3);
          applyCellValue(value, true);
        };
        const onUp = () => {
          window.removeEventListener("mousemove", onMove);
          window.removeEventListener("mouseup", onUp);
          input.value = appState.matrixDraft[i][j].toFixed(3);
        };
        window.addEventListener("mousemove", onMove);
        window.addEventListener("mouseup", onUp, { once: true });
      });
      matrixGrid.appendChild(input);
    }
  }
}

function buildUI() {
  panel.innerHTML = "";
  statsNodes = {};

  if (appState.values.show_hud !== false) {
    createSection("Info", (body) => {
      ["Graphics FPS", "Physics FPS", "Speed ratio", "Particles", "Types", "Per-type counts", "Velocity std dev", "Matrix version"].forEach((k) => {
        const line = document.createElement("div");
        line.className = "stats-line";
        const name = document.createElement("span");
        const value = document.createElement("span");
        name.textContent = k;
        value.textContent = "--";
        line.append(name, value);
        body.appendChild(line);
        statsNodes[k] = value;
      });
    });
  }

  createSection("Controls", (body) => {
    const runRow = document.createElement("div");
    runRow.className = "row";
    const pauseBtn = document.createElement("button");
    pauseBtn.textContent = appState.paused ? "Resume" : "Pause";
    pauseBtn.addEventListener("click", async () => { await setPaused(!appState.paused); });
    const resetBtn = document.createElement("button");
    resetBtn.textContent = "Reset";
    resetBtn.addEventListener("click", async () => setRemoteState(await postJSON("/api/config/reset")));
    const randomizeBtn = document.createElement("button");
    randomizeBtn.textContent = "Randomize Seed";
    randomizeBtn.addEventListener("click", async () => setRemoteState(await postJSON("/api/config/randomize")));
    runRow.append(pauseBtn, resetBtn, randomizeBtn);
    body.append(runRow);

    const presetRow = document.createElement("div");
    presetRow.className = "row";
    const presetSelect = document.createElement("select");
    appState.presets.forEach((name) => {
      const o = document.createElement("option");
      o.value = name;
      o.textContent = name;
      presetSelect.appendChild(o);
    });
    const presetBtn = document.createElement("button");
    presetBtn.textContent = "Load preset";
    presetBtn.addEventListener("click", async () => setRemoteState(await postJSON("/api/config/preset", { name: presetSelect.value })));
    presetRow.append(presetSelect, presetBtn);
    body.appendChild(presetRow);
  });

  appState.sections.forEach((section) => {
    createSection(section.label, (body) => {
      section.controls.forEach((control) => bindControl(body, control));
      if (section.key === "simulation") {
        buildParticleCountControls(body);
      }
    });
    if (section.key === "simulation") {
      createSection("Interaction Matrix", (body) => buildMatrixEditor(body));
    }
  });
  syncMatrix();
}

function validateParams() {
  const n = Number(appState.values.species_count || 0);
  const matrix = appState.values.interaction_matrix || [];
  if (matrix.length !== n || matrix.some((row) => row.length !== n)) console.assert(false, "matrix dimensions must match species_count");
  if (DEV_MODE) {
    matrix.forEach((row) => row.forEach((v) => console.assert(Number.isFinite(v) && v >= -1 && v <= 1, "matrix value out of bounds", v)));
  }
}

function drawInstances(offsetX, offsetY) {
  const shifted = new Float32Array(positions.length);
  for (let i = 0; i < pointCount; i += 1) {
    shifted[i * 2] = positions[i * 2] + offsetX;
    shifted[i * 2 + 1] = positions[i * 2 + 1] + offsetY;
  }
  gl.bindBuffer(gl.ARRAY_BUFFER, posBuffer);
  gl.bufferData(gl.ARRAY_BUFFER, shifted, gl.DYNAMIC_DRAW);
  gl.enableVertexAttribArray(aPos);
  gl.vertexAttribPointer(aPos, 2, gl.FLOAT, false, 0, 0);

  gl.bindBuffer(gl.ARRAY_BUFFER, speciesBuffer);
  gl.bufferData(gl.ARRAY_BUFFER, species, gl.DYNAMIC_DRAW);
  gl.enableVertexAttribArray(aSpecies);
  gl.vertexAttribPointer(aSpecies, 1, gl.FLOAT, false, 0, 0);

  gl.bindBuffer(gl.ARRAY_BUFFER, velBuffer);
  gl.bufferData(gl.ARRAY_BUFFER, velocities, gl.DYNAMIC_DRAW);
  gl.enableVertexAttribArray(aVel);
  gl.vertexAttribPointer(aVel, 2, gl.FLOAT, false, 0, 0);
  gl.drawArrays(gl.POINTS, 0, pointCount);
}

function draw() {
  const bg = Number(appState.values.background_alpha ?? 1);
  gl.clearColor(0.02, 0.03, 0.08, bg);
  gl.clear(gl.COLOR_BUFFER_BIT);
  if (!pointCount) return;

  gl.useProgram(program);
  gl.uniform2f(uScale, window.devicePixelRatio || 1, 1);
  gl.uniform1f(uPointSize, Number(appState.values.point_size ?? 3));
  gl.uniform1f(uOpacity, Number(appState.values.point_opacity ?? 0.95));
  const mode = appState.values.color_mode === "velocity" ? 1 : appState.values.color_mode === "mono" ? 2 : 0;
  gl.uniform1i(uColorMode, mode);

  if (appState.values.pbc_tiling) {
    for (let dy = -1; dy <= 1; dy += 1) for (let dx = -1; dx <= 1; dx += 1) drawInstances(dx, dy);
  } else {
    drawInstances(0, 0);
  }
}

function consumeFrame(buffer) {
  const data = new Float32Array(buffer);
  pointCount = Math.floor(data.length / 5);
  positions = new Float32Array(pointCount * 2);
  species = new Float32Array(pointCount);
  velocities = new Float32Array(pointCount * 2);
  for (let i = 0; i < pointCount; i += 1) {
    positions[i * 2] = data[i * 5];
    positions[i * 2 + 1] = data[i * 5 + 1];
    species[i] = data[i * 5 + 2];
    velocities[i * 2] = data[i * 5 + 3];
    velocities[i * 2 + 1] = data[i * 5 + 4];
  }
  perf.physicsFrames += 1;
}

function velocityStdDev() {
  if (!pointCount) return 0;
  let sum = 0;
  for (let i = 0; i < velocities.length; i += 1) sum += velocities[i] * velocities[i];
  return Math.sqrt(sum / velocities.length);
}

function updateStats() {
  const now = performance.now();
  perf.gfxFrames += 1;
  if (now - perf.lastStamp >= 500) {
    const scale = 1000 / (now - perf.lastStamp);
    perf.gfxFps = perf.gfxFrames * scale;
    perf.physicsFps = perf.physicsFrames * scale;
    perf.gfxFrames = 0;
    perf.physicsFrames = 0;
    perf.lastStamp = now;
    if (statsNodes["Graphics FPS"]) {
      statsNodes["Graphics FPS"].textContent = perf.gfxFps.toFixed(1);
      statsNodes["Physics FPS"].textContent = perf.physicsFps.toFixed(1);
      statsNodes["Speed ratio"].textContent = (perf.physicsFps / Math.max(perf.gfxFps, 0.1)).toFixed(2);
      statsNodes["Particles"].textContent = String(pointCount);
      statsNodes["Types"].textContent = String(appState.values.species_count ?? "--");
      const displayCounts = Array.isArray(appState.values.particle_counts) ? appState.values.particle_counts.join(",") : "--";
      statsNodes["Per-type counts"].textContent = displayCounts;
      statsNodes["Velocity std dev"].textContent = velocityStdDev().toFixed(4);
      statsNodes["Matrix version"].textContent = String(appState.values.matrix_version ?? "--");
    }
  }
}

function animate() {
  draw();
  updateStats();
  requestAnimationFrame(animate);
}

async function init() {
  const res = await fetch("/api/config");
  const data = await res.json();
  setRemoteState(data);
  appState.subscribers.add(() => buildUI());
  buildUI();
  animate();
}

const wsScheme = window.location.protocol === "https:" ? "wss" : "ws";
const ws = new WebSocket(`${wsScheme}://${window.location.host}/ws`);
ws.binaryType = "arraybuffer";
ws.onmessage = (event) => consumeFrame(event.data);

init();
