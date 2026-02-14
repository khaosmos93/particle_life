const canvas = document.getElementById("canvas");
const gl = canvas.getContext("webgl");
if (!gl) throw new Error("WebGL unavailable");

const ui = {
  sections: document.getElementById("sections"),
  presetSelect: document.getElementById("presetSelect"),
  applyPresetBtn: document.getElementById("applyPresetBtn"),
  resetBtn: document.getElementById("resetBtn"),
  randomBtn: document.getElementById("randomBtn"),
  hud: document.getElementById("hud"),
};

let configSchema = null;
let configValues = null;
const controls = {};
const panelState = new Map();
let matrixInput = null;
let matrixValueSpans = [];

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
}
`;

const fragmentSrc = `
precision mediump float;
varying vec4 v_color;
uniform float u_opacity;
void main() {
  vec2 c = gl_PointCoord - vec2(0.5);
  if (dot(c, c) > 0.25) discard;
  gl_FragColor = vec4(v_color.rgb, u_opacity);
}
`;

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

let pointCount = 0;
let positions = new Float32Array();
let species = new Float32Array();
let velocities = new Float32Array();

function resize() {
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.floor(window.innerWidth * dpr);
  canvas.height = Math.floor(window.innerHeight * dpr);
  gl.viewport(0, 0, canvas.width, canvas.height);
}
window.addEventListener("resize", resize);
resize();

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
  gl.clearColor(0.02, 0.03, 0.08, bg);
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
  ui.hud.style.display = configValues?.show_hud ? "block" : "none";
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
  draw();
}

async function postJSON(path, body = {}) {
  const r = await fetch(path, { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(body) });
  return r.json();
}

function valueText(v) {
  return Number.isInteger(v) ? String(v) : Number(v).toFixed(3).replace(/0+$/, "").replace(/\.$/, "");
}

function buildControl(control) {
  const row = document.createElement("label");
  row.innerHTML = `<span class="left">${control.label}</span>`;
  let input;
  if (control.type === "select") {
    input = document.createElement("select");
    control.options.forEach((opt) => {
      const o = document.createElement("option");
      o.value = opt; o.textContent = opt; input.appendChild(o);
    });
  } else {
    input = document.createElement("input");
    input.type = control.type === "toggle" ? "checkbox" : control.type;
    if (control.type !== "toggle") {
      if (control.min !== undefined) input.min = control.min;
      if (control.max !== undefined) input.max = control.max;
      if (control.step !== undefined) input.step = control.step;
    }
  }
  input.className = "ctrl";
  const val = document.createElement("span");
  val.className = "val";
  row.appendChild(input);
  row.appendChild(val);
  controls[control.key] = { input, val, control };

  input.addEventListener("input", async () => {
    const raw = control.type === "toggle" ? input.checked : input.value;
    const parsed = control.type === "range" || control.type === "number" ? Number(raw) : raw;
    const result = await postJSON("/api/config/update", { updates: { [control.key]: parsed } });
    configValues = result.values;
    syncValues();
  });
  return row;
}

function toggleSection(sectionKey, header, body) {
  const collapsed = !panelState.get(sectionKey);
  panelState.set(sectionKey, collapsed);
  body.style.display = collapsed ? "none" : "block";
  header.dataset.collapsed = String(collapsed);
}

function buildCollapsibleSection(section) {
  const box = document.createElement("div");
  box.className = "section";
  const header = document.createElement("h3");
  header.className = "section-header";
  header.textContent = section.label;
  const body = document.createElement("div");
  body.className = "section-body";

  const initiallyCollapsed = panelState.get(section.key) ?? false;
  panelState.set(section.key, initiallyCollapsed);
  if (initiallyCollapsed) {
    body.style.display = "none";
    header.dataset.collapsed = "true";
  }

  header.addEventListener("click", () => toggleSection(section.key, header, body));
  section.controls.forEach((c) => body.appendChild(buildControl(c)));
  box.appendChild(header);
  box.appendChild(body);
  return box;
}

function cloneMatrix(matrix) {
  return matrix.map((row) => row.map((v) => Number(v)));
}

function clampMatrixValue(value) {
  return Math.max(-1, Math.min(1, Number(value)));
}

async function pushMatrix(matrix) {
  const next = cloneMatrix(matrix);
  const result = await postJSON("/api/config/update", { updates: { interaction_matrix: next } });
  configValues = result.values;
  syncValues();
}

function matrixMutate(mutator) {
  if (!configValues?.interaction_matrix) return;
  const next = cloneMatrix(configValues.interaction_matrix);
  mutator(next);
  pushMatrix(next);
}

function normalizeMatrixRows(matrix) {
  return matrix.map((row) => {
    const maxAbs = row.reduce((acc, v) => Math.max(acc, Math.abs(v)), 0);
    if (maxAbs < 1e-9) return row.slice();
    return row.map((v) => clampMatrixValue(v / maxAbs));
  });
}

function buildMatrixEditor() {
  const wrapper = document.createElement("div");
  wrapper.className = "matrix-editor";

  const toolbar = document.createElement("div");
  toolbar.className = "row";

  const randomizeBtn = document.createElement("button");
  randomizeBtn.textContent = "Randomize Matrix";
  randomizeBtn.addEventListener("click", () => {
    matrixMutate((matrix) => {
      for (let i = 0; i < matrix.length; i += 1) {
        for (let j = 0; j < matrix.length; j += 1) {
          matrix[i][j] = i === j ? 0.2 + Math.random() * 0.8 : -1 + Math.random() * 2;
        }
      }
    });
  });

  const resetBtn = document.createElement("button");
  resetBtn.textContent = "Reset Matrix";
  resetBtn.addEventListener("click", () => {
    const n = Number(configValues?.species_count ?? 0);
    matrixMutate((matrix) => {
      for (let i = 0; i < n; i += 1) {
        for (let j = 0; j < n; j += 1) matrix[i][j] = i === j ? 1 : 0;
      }
    });
  });

  const symmetryBtn = document.createElement("button");
  symmetryBtn.textContent = "Symmetrize";
  symmetryBtn.addEventListener("click", () => {
    matrixMutate((matrix) => {
      for (let i = 0; i < matrix.length; i += 1) {
        for (let j = i + 1; j < matrix.length; j += 1) {
          const avg = clampMatrixValue((matrix[i][j] + matrix[j][i]) * 0.5);
          matrix[i][j] = avg;
          matrix[j][i] = avg;
        }
      }
    });
  });

  const normalizeBtn = document.createElement("button");
  normalizeBtn.textContent = "Normalize";
  normalizeBtn.addEventListener("click", () => {
    matrixMutate((matrix) => {
      const normalized = normalizeMatrixRows(matrix);
      for (let i = 0; i < matrix.length; i += 1) matrix[i] = normalized[i];
    });
  });

  toolbar.append(randomizeBtn, resetBtn, symmetryBtn, normalizeBtn);
  wrapper.appendChild(toolbar);

  matrixInput = document.createElement("div");
  matrixInput.className = "matrix-grid";
  wrapper.appendChild(matrixInput);
  return wrapper;
}

function wireMatrixDrag(input, i, j) {
  let startY = 0;
  let startValue = 0;
  const onMove = (event) => {
    const delta = (startY - event.clientY) * 0.01;
    const next = clampMatrixValue(startValue + delta);
    input.value = String(next);
    input.dispatchEvent(new Event("change"));
  };
  const onUp = () => {
    window.removeEventListener("mousemove", onMove);
    window.removeEventListener("mouseup", onUp);
  };
  input.addEventListener("mousedown", (event) => {
    if (event.button !== 0) return;
    event.preventDefault();
    startY = event.clientY;
    startValue = Number(configValues.interaction_matrix[i][j]);
    window.addEventListener("mousemove", onMove);
    window.addEventListener("mouseup", onUp);
  });
}

function syncMatrixEditor() {
  if (!matrixInput) return;
  const matrix = configValues?.interaction_matrix;
  if (!matrix) {
    matrixInput.innerHTML = "";
    return;
  }
  const n = matrix.length;
  matrixInput.style.gridTemplateColumns = `repeat(${n}, minmax(42px, 1fr))`;

  if (matrixValueSpans.length !== n * n) {
    matrixInput.innerHTML = "";
    matrixValueSpans = [];
    for (let i = 0; i < n; i += 1) {
      for (let j = 0; j < n; j += 1) {
        const input = document.createElement("input");
        input.type = "number";
        input.min = "-1";
        input.max = "1";
        input.step = "0.01";
        input.className = "matrix-cell";
        input.addEventListener("change", async () => {
          const next = cloneMatrix(configValues.interaction_matrix);
          next[i][j] = clampMatrixValue(input.value);
          await pushMatrix(next);
        });
        wireMatrixDrag(input, i, j);
        matrixValueSpans.push(input);
        matrixInput.appendChild(input);
      }
    }
  }

  matrixValueSpans.forEach((input, idx) => {
    const i = Math.floor(idx / n);
    const j = idx % n;
    input.value = valueText(matrix[i][j]);
    input.title = `${i}→${j}`;
  });
}

function syncValues() {
  Object.entries(controls).forEach(([key, ctx]) => {
    const v = configValues[key];
    if (ctx.control.type === "toggle") ctx.input.checked = Boolean(v);
    else ctx.input.value = v;
    ctx.val.textContent = ctx.control.type === "toggle" ? (v ? "on" : "off") : valueText(v);
  });
  syncMatrixEditor();
  draw();
}

async function initUI() {
  const res = await fetch("/api/config");
  const data = await res.json();
  configSchema = data.sections;
  configValues = data.values;

  data.presets.forEach((name) => {
    const o = document.createElement("option");
    o.value = name; o.textContent = name; ui.presetSelect.appendChild(o);
  });

  configSchema.forEach((section) => {
    ui.sections.appendChild(buildCollapsibleSection(section));
  });

  const matrixSection = buildCollapsibleSection({ key: "interaction_matrix", label: "Interaction Matrix", controls: [] });
  matrixSection.querySelector(".section-body").appendChild(buildMatrixEditor());
  ui.sections.appendChild(matrixSection);

  ui.resetBtn.addEventListener("click", async () => {
    const r = await postJSON("/api/config/reset");
    configValues = r.values; syncValues();
  });
  ui.randomBtn.addEventListener("click", async () => {
    const r = await postJSON("/api/config/randomize");
    configValues = r.values; syncValues();
  });
  ui.applyPresetBtn.addEventListener("click", async () => {
    const r = await postJSON("/api/config/preset", { name: ui.presetSelect.value });
    configValues = r.values; syncValues();
  });

  syncValues();
}

const wsScheme = window.location.protocol === "https:" ? "wss" : "ws";
const ws = new WebSocket(`${wsScheme}://${window.location.host}/ws`);
ws.binaryType = "arraybuffer";
ws.onmessage = (event) => consumeFrame(event.data);

initUI();
