const canvas = document.getElementById("canvas");
const panel = document.getElementById("panel");
const gl = canvas.getContext("webgl");
if (!gl) throw new Error("WebGL unavailable");

const TYPE_COLORS = ["#ff6f5f", "#56c3ff", "#7aff63", "#ffe26a", "#d086ff", "#ffa04d", "#60ffd0", "#c8dbff", "#ff73b8", "#88ff9e", "#6f8bff", "#ffc56f"];

const uiState = {
  sectionCollapsed: new Map(),
  extras: {
    pause: false,
    accelerator: "CPU",
    threads: 1,
    fixed_step: false,
    clear_screen: true,
    shader: "default",
    palette: "species",
    matrixPreset: "fully_random",
    positionsPreset: "random",
    typesPreset: "equal",
  },
};

let configValues = null;
let pointCount = 0;
let positions = new Float32Array();
let species = new Float32Array();
let velocities = new Float32Array();
let matrixGrid;
let statsNodes = {};

const perf = {
  gfxFrames: 0,
  physicsFrames: 0,
  lastStamp: performance.now(),
  gfxFps: 0,
  physicsFps: 0,
};

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

function valueText(v) {
  return Number.isInteger(v) ? String(v) : Number(v).toFixed(3).replace(/0+$/, "").replace(/\.$/, "");
}

function clampMatrixValue(value) {
  return Math.max(-1, Math.min(1, Number(value)));
}

async function postJSON(path, body = {}) {
  const r = await fetch(path, { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(body) });
  return r.json();
}

async function applyUpdate(key, value) {
  if (!configValues) return;
  const result = await postJSON("/api/config/update", { updates: { [key]: value } });
  configValues = result.values;
  syncUI();
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

  const collapsed = uiState.sectionCollapsed.get(key) ?? false;
  if (collapsed) {
    header.dataset.collapsed = "true";
    body.style.display = "none";
    chevron.textContent = "▸";
  }
  header.addEventListener("click", () => {
    const nowCollapsed = body.style.display !== "none";
    uiState.sectionCollapsed.set(key, nowCollapsed);
    body.style.display = nowCollapsed ? "none" : "block";
    header.dataset.collapsed = nowCollapsed ? "true" : "false";
    chevron.textContent = nowCollapsed ? "▸" : "▾";
  });

  section.append(header, body);
  panel.appendChild(section);
}

function bindRange(parent, cfg) {
  const wrap = document.createElement("div");
  wrap.className = "field";
  const row = document.createElement("div");
  row.className = "field-row";
  const label = document.createElement("label");
  label.textContent = cfg.label;
  const input = document.createElement("input");
  input.type = "range";
  input.min = cfg.min;
  input.max = cfg.max;
  input.step = cfg.step;
  input.value = configValues[cfg.key] ?? cfg.default;
  const value = document.createElement("span");
  value.className = "value";
  value.textContent = valueText(input.value);
  const applyTag = document.createElement("span");
  applyTag.className = "apply-tag";
  applyTag.textContent = cfg.apply || "";
  input.addEventListener("input", () => {
    value.textContent = valueText(input.value);
  });
  input.addEventListener("change", () => applyUpdate(cfg.key, Number(input.value)));
  row.append(label, input, value, applyTag);
  wrap.appendChild(row);
  parent.appendChild(wrap);
}

function bindSelect(parent, cfg) {
  const row = document.createElement("div");
  row.className = "field-row";
  const label = document.createElement("label");
  label.textContent = cfg.label;
  const select = document.createElement("select");
  cfg.options.forEach((opt) => {
    const o = document.createElement("option");
    o.value = opt;
    o.textContent = opt;
    select.appendChild(o);
  });
  select.value = String(configValues[cfg.key] ?? cfg.default);
  const value = document.createElement("span");
  value.className = "value";
  value.textContent = select.value;
  const applyTag = document.createElement("span");
  applyTag.className = "apply-tag";
  applyTag.textContent = cfg.apply || "";
  select.addEventListener("change", () => {
    value.textContent = select.value;
    applyUpdate(cfg.key, select.value);
  });
  row.append(label, select, value, applyTag);
  parent.appendChild(row);
}

function bindToggle(parent, cfg) {
  const row = document.createElement("div");
  row.className = "field-row";
  const label = document.createElement("label");
  label.textContent = cfg.label;
  const input = document.createElement("input");
  input.type = "checkbox";
  input.checked = Boolean(configValues[cfg.key] ?? cfg.default);
  const value = document.createElement("span");
  value.className = "value";
  value.textContent = input.checked ? "on" : "off";
  const applyTag = document.createElement("span");
  applyTag.className = "apply-tag";
  applyTag.textContent = cfg.apply || "";
  input.addEventListener("change", () => {
    value.textContent = input.checked ? "on" : "off";
    applyUpdate(cfg.key, input.checked);
  });
  row.append(label, input, value, applyTag);
  parent.appendChild(row);
}

function createControl(parent, cfg) {
  if (cfg.type === "range") bindRange(parent, cfg);
  if (cfg.type === "select") bindSelect(parent, cfg);
  if (cfg.type === "toggle") bindToggle(parent, cfg);
}

function matrixColor(v) {
  const val = clampMatrixValue(v);
  const intensity = Math.floor(Math.abs(val) * 180) + 35;
  if (val >= 0) return `rgb(${30}, ${intensity}, ${35})`;
  return `rgb(${intensity}, ${32}, ${32})`;
}

async function pushMatrix(next) {
  const result = await postJSON("/api/config/update", { updates: { interaction_matrix: next } });
  configValues = result.values;
  syncMatrix();
}

function cloneMatrix() {
  return configValues.interaction_matrix.map((row) => row.map((v) => Number(v)));
}

function buildMatrixEditor(parent) {
  const toolbar = document.createElement("div");
  toolbar.className = "matrix-toolbar";

  const presetSelect = document.createElement("select");
  [
    ["fully_random", "fully random"],
    ["zero", "all zeros"],
    ["identity", "identity +"],
    ["symmetric_random", "symmetric random"],
  ].forEach(([value, label]) => {
    const o = document.createElement("option");
    o.value = value;
    o.textContent = label;
    presetSelect.appendChild(o);
  });
  presetSelect.value = uiState.extras.matrixPreset;

  const presetApply = document.createElement("button");
  presetApply.textContent = "Apply";
  presetApply.addEventListener("click", async () => {
    const matrix = cloneMatrix();
    const n = matrix.length;
    if (presetSelect.value === "fully_random") {
      for (let i = 0; i < n; i += 1) for (let j = 0; j < n; j += 1) matrix[i][j] = -1 + Math.random() * 2;
    }
    if (presetSelect.value === "zero") {
      for (let i = 0; i < n; i += 1) for (let j = 0; j < n; j += 1) matrix[i][j] = 0;
    }
    if (presetSelect.value === "identity") {
      for (let i = 0; i < n; i += 1) for (let j = 0; j < n; j += 1) matrix[i][j] = i === j ? 1 : 0;
    }
    if (presetSelect.value === "symmetric_random") {
      for (let i = 0; i < n; i += 1) {
        for (let j = i; j < n; j += 1) {
          const value = -1 + Math.random() * 2;
          matrix[i][j] = value;
          matrix[j][i] = value;
        }
      }
    }
    await pushMatrix(matrix);
  });
  toolbar.append(presetSelect, presetApply);

  const clipboardRow = document.createElement("div");
  clipboardRow.className = "row";
  const copyBtn = document.createElement("button");
  copyBtn.textContent = "Copy";
  copyBtn.addEventListener("click", async () => {
    const text = configValues.interaction_matrix.map((r) => r.map((v) => Number(v).toFixed(4)).join(" ")).join("\n");
    await navigator.clipboard.writeText(text);
  });
  const pasteBtn = document.createElement("button");
  pasteBtn.textContent = "Paste";
  pasteBtn.addEventListener("click", async () => {
    const text = await navigator.clipboard.readText();
    const rows = text.trim().split(/\n+/).map((line) => line.trim().split(/[\s,;]+/).map(Number));
    const n = configValues.interaction_matrix.length;
    if (rows.length !== n || rows.some((r) => r.length !== n || r.some((v) => Number.isNaN(v)))) return;
    await pushMatrix(rows.map((r) => r.map((v) => clampMatrixValue(v))));
  });
  clipboardRow.append(copyBtn, pasteBtn);

  matrixGrid = document.createElement("div");
  matrixGrid.className = "matrix-grid";

  parent.append(toolbar, clipboardRow, matrixGrid);
}

function syncMatrix() {
  if (!matrixGrid || !configValues?.interaction_matrix) return;
  const matrix = configValues.interaction_matrix;
  const n = matrix.length;
  matrixGrid.innerHTML = "";
  matrixGrid.style.gridTemplateColumns = `16px repeat(${n}, 24px)`;

  const corner = document.createElement("div");
  matrixGrid.appendChild(corner);
  for (let c = 0; c < n; c += 1) {
    const dot = document.createElement("div");
    dot.className = "matrix-axis-dot";
    dot.style.background = TYPE_COLORS[c % TYPE_COLORS.length];
    matrixGrid.appendChild(dot);
  }

  let dragging = null;
  const applyDelta = (event, i, j) => {
    if (!dragging) return;
    const matrixNext = cloneMatrix();
    const delta = ((dragging.startY - event.clientY) + (event.clientX - dragging.startX)) * 0.004;
    matrixNext[i][j] = clampMatrixValue(dragging.startValue + delta);
    configValues.interaction_matrix = matrixNext;
    syncMatrix();
  };

  for (let i = 0; i < n; i += 1) {
    const rowDot = document.createElement("div");
    rowDot.className = "matrix-axis-dot";
    rowDot.style.background = TYPE_COLORS[i % TYPE_COLORS.length];
    matrixGrid.appendChild(rowDot);

    for (let j = 0; j < n; j += 1) {
      const cell = document.createElement("div");
      const value = Number(matrix[i][j]);
      cell.className = "matrix-cell";
      cell.style.background = matrixColor(value);
      cell.title = `${i} → ${j}: ${value.toFixed(3)}`;
      cell.addEventListener("mousedown", (event) => {
        if (event.button !== 0) return;
        dragging = { i, j, startY: event.clientY, startX: event.clientX, startValue: value };
        const onMove = (moveEvent) => applyDelta(moveEvent, i, j);
        const onUp = async () => {
          window.removeEventListener("mousemove", onMove);
          window.removeEventListener("mouseup", onUp);
          const latest = cloneMatrix();
          dragging = null;
          await pushMatrix(latest);
        };
        window.addEventListener("mousemove", onMove);
        window.addEventListener("mouseup", onUp, { once: true });
      });
      matrixGrid.appendChild(cell);
    }
  }
}

function buildUI() {
  panel.innerHTML = "";
  createSection("Info", (body) => {
    const keys = ["Graphics FPS", "Physics FPS", "Speed ratio", "Particles", "Types", "Velocity std dev"];
    keys.forEach((k) => {
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

  createSection("Particles", (body) => {
    const list = [
      { key: "particles_per_species", type: "range", label: "Particle count/type", min: 20, max: 400, step: 10, default: 160, apply: "reset" },
      { key: "species_count", type: "range", label: "Types", min: 2, max: 12, step: 1, default: 6, apply: "reset" },
    ];
    list.forEach((cfg) => createControl(body, cfg));

    const eqRow = document.createElement("div");
    eqRow.className = "field-row";
    eqRow.innerHTML = "<label>Equalize type count</label>";
    const eqBtn = document.createElement("button");
    eqBtn.textContent = "Apply";
    eqBtn.addEventListener("click", () => applyUpdate("particles_per_species", configValues.particles_per_species));
    eqRow.appendChild(eqBtn);
    body.appendChild(eqRow);
  });

  createSection("Interaction matrix", (body) => buildMatrixEditor(body));

  createSection("Presets", (body) => {
    const mini = document.createElement("div");
    mini.className = "mini";
    mini.textContent = "Matrix / positions / types";
    body.appendChild(mini);
    const matrixPreset = document.createElement("select");
    ["Default", "Dense", "Sparse", "Chaotic"].forEach((name) => {
      const o = document.createElement("option");
      o.value = name;
      o.textContent = name;
      matrixPreset.appendChild(o);
    });
    const applyBtn = document.createElement("button");
    applyBtn.textContent = "Load";
    applyBtn.addEventListener("click", async () => {
      const response = await postJSON("/api/config/preset", { name: matrixPreset.value });
      configValues = response.values;
      syncUI();
    });
    const row = document.createElement("div");
    row.className = "row";
    row.append(matrixPreset, applyBtn);
    body.appendChild(row);

    const placeholders = [
      ["positions", ["random", "ring", "uniform"]],
      ["types", ["equal", "clusters", "random"]],
    ];
    placeholders.forEach(([name, options]) => {
      const r = document.createElement("div");
      r.className = "row";
      const s = document.createElement("select");
      options.forEach((opt) => {
        const o = document.createElement("option"); o.value = opt; o.textContent = `${name}: ${opt}`; s.appendChild(o);
      });
      s.value = uiState.extras[`${name}Preset`];
      s.addEventListener("change", () => {
        uiState.extras[`${name}Preset`] = s.value;
        applyUpdate(`${name}_preset`, s.value);
      });
      r.appendChild(s);
      body.appendChild(r);
    });
  });

  createSection("Physics", (body) => {
    [
      { key: "dt", type: "range", label: "fixed step", min: 0.001, max: 0.06, step: 0.001, default: 0.015, apply: "immediate" },
      { key: "interaction_radius", type: "range", label: "rmax", min: 0.02, max: 0.35, step: 0.005, default: 0.11, apply: "immediate" },
      { key: "damping", type: "range", label: "friction", min: 0.85, max: 0.999, step: 0.001, default: 0.975, apply: "immediate" },
      { key: "force_scale", type: "range", label: "force", min: 0.05, max: 1.2, step: 0.01, default: 0.42, apply: "immediate" },
      { key: "boundary_mode", type: "select", label: "wrap", options: ["wrap", "bounce"], default: "wrap", apply: "immediate" },
      { key: "steps_per_frame", type: "range", label: "accelerator", min: 1, max: 8, step: 1, default: 1, apply: "immediate" },
    ].forEach((cfg) => createControl(body, cfg));

    const pauseRow = document.createElement("div");
    pauseRow.className = "field-row";
    pauseRow.innerHTML = "<label>Pause</label>";
    const pause = document.createElement("input");
    pause.type = "checkbox";
    pause.checked = uiState.extras.pause;
    pause.addEventListener("change", () => { uiState.extras.pause = pause.checked; });
    pauseRow.appendChild(pause);
    body.appendChild(pauseRow);

    const threadRow = document.createElement("div");
    threadRow.className = "field-row";
    threadRow.innerHTML = "<label>threads</label>";
    const threads = document.createElement("input");
    threads.type = "number";
    threads.min = "1";
    threads.max = "32";
    threads.value = String(uiState.extras.threads);
    threads.addEventListener("change", () => {
      uiState.extras.threads = Number(threads.value);
      applyUpdate("threads", uiState.extras.threads);
    });
    threadRow.appendChild(threads);
    body.appendChild(threadRow);
  });

  createSection("Graphics", (body) => {
    [
      { key: "point_size", type: "range", label: "particle size", min: 1, max: 8, step: 0.1, default: 3, apply: "immediate" },
      { key: "color_mode", type: "select", label: "palette", options: ["species", "velocity", "mono"], default: "species", apply: "immediate" },
      { key: "background_alpha", type: "range", label: "clear screen", min: 0.02, max: 1.0, step: 0.01, default: 1, apply: "immediate" },
      { key: "pbc_tiling", type: "toggle", label: "shader: 3x3 PBC", default: false, apply: "immediate" },
    ].forEach((cfg) => createControl(body, cfg));
  });

  syncMatrix();
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
  const bg = Number(configValues?.background_alpha ?? 1.0);
  gl.clearColor(0.02, 0.03, 0.08, uiState.extras.clear_screen ? bg : 0);
  gl.clear(gl.COLOR_BUFFER_BIT);
  if (!pointCount) return;

  gl.useProgram(program);
  gl.uniform2f(uScale, window.devicePixelRatio || 1, 1);
  gl.uniform1f(uPointSize, Number(configValues?.point_size ?? 3));
  gl.uniform1f(uOpacity, Number(configValues?.point_opacity ?? 0.95));
  const mode = configValues?.color_mode === "velocity" ? 1 : configValues?.color_mode === "mono" ? 2 : 0;
  gl.uniform1i(uColorMode, mode);

  if (configValues?.pbc_tiling) {
    for (let dy = -1; dy <= 1; dy += 1) for (let dx = -1; dx <= 1; dx += 1) drawInstances(dx, dy);
  } else {
    drawInstances(0, 0);
  }
}

function consumeFrame(buffer) {
  if (uiState.extras.pause) return;
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
      statsNodes["Types"].textContent = String(configValues?.species_count ?? "--");
      statsNodes["Velocity std dev"].textContent = velocityStdDev().toFixed(4);
    }
  }
}

function animate() {
  draw();
  updateStats();
  requestAnimationFrame(animate);
}

function syncUI() {
  buildUI();
}

async function init() {
  try {
    const res = await fetch("/api/config");
    const data = await res.json();
    configValues = data.values;
  } catch (_error) {
    configValues = {
      species_count: 6,
      particles_per_species: 160,
      interaction_radius: 0.11,
      damping: 0.975,
      force_scale: 0.42,
      dt: 0.015,
      boundary_mode: "wrap",
      steps_per_frame: 1,
      point_size: 3,
      point_opacity: 0.95,
      background_alpha: 1,
      pbc_tiling: false,
      color_mode: "species",
      interaction_matrix: Array.from({ length: 6 }, (_, i) => Array.from({ length: 6 }, (_, j) => (i === j ? 1 : 0))),
    };
  }
  buildUI();
  animate();
}

const wsScheme = window.location.protocol === "https:" ? "wss" : "ws";
const ws = new WebSocket(`${wsScheme}://${window.location.host}/ws`);
ws.binaryType = "arraybuffer";
ws.onmessage = (event) => consumeFrame(event.data);

init();
