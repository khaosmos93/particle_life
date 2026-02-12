const canvas = document.getElementById("glcanvas");
const gl = canvas.getContext("webgl");

if (!gl) {
  throw new Error("WebGL not supported");
}

const protocol = location.protocol === "https:" ? "wss" : "ws";
const ws = new WebSocket(`${protocol}://${location.host}/ws`);
ws.binaryType = "arraybuffer";

const dtSlider = document.getElementById("dtSlider");
const speedSlider = document.getElementById("speedSlider");
const sendEverySlider = document.getElementById("sendEverySlider");
const pointSizeSlider = document.getElementById("pointSize");
const seedInput = document.getElementById("seedInput");
const colorSchemeSelect = document.getElementById("colorSchemeSelect");

const dtValue = document.getElementById("dtValue");
const speedValue = document.getElementById("speedValue");
const sendEveryValue = document.getElementById("sendEveryValue");
const pointSizeValue = document.getElementById("pointSizeValue");

const fpsValue = document.getElementById("fpsValue");
const realtimeValue = document.getElementById("realtimeValue");
const physicsValue = document.getElementById("physicsValue");

let wsStartTime = performance.now();
let fpsWindowStart = performance.now();
let fpsFrameCount = 0;
let latestFps = 0;
let physicsTime = 0;
let currentColorScheme = colorSchemeSelect.value;

function clamp01(x) {
  return Math.min(1, Math.max(0, x));
}

function stateToRgb(state, scheme) {
  const x = state[0] ?? 0;
  const y = state[1] ?? 0;
  const z = state[2] ?? 0;

  if (scheme === "normalize") {
    const norm = Math.sqrt(x * x + y * y + z * z) || 1;
    return [clamp01(0.5 + 0.5 * x / norm), clamp01(0.5 + 0.5 * y / norm), clamp01(0.5 + 0.5 * z / norm)];
  }

  if (scheme === "abs") {
    const sum = Math.abs(x) + Math.abs(y) + Math.abs(z) || 1;
    return [Math.abs(x) / sum, Math.abs(y) / sum, Math.abs(z) / sum];
  }

  if (scheme === "softmax") {
    const ex = Math.exp(Math.max(-10, Math.min(10, x)));
    const ey = Math.exp(Math.max(-10, Math.min(10, y)));
    const ez = Math.exp(Math.max(-10, Math.min(10, z)));
    const denom = ex + ey + ez || 1;
    return [ex / denom, ey / denom, ez / denom];
  }

  if (scheme === "hsv_like") {
    const norm = Math.sqrt(x * x + y * y + z * z) || 1;
    const nx = 0.5 + 0.5 * x / norm;
    const ny = 0.5 + 0.5 * y / norm;
    const nz = 0.5 + 0.5 * z / norm;
    const hue = (Math.atan2(ny - 0.5, nx - 0.5) / (2 * Math.PI) + 1) % 1;
    const sat = clamp01(Math.hypot(nx - 0.5, ny - 0.5) * 2);
    const val = clamp01(nz);
    const i = Math.floor(hue * 6);
    const f = hue * 6 - i;
    const p = val * (1 - sat);
    const q = val * (1 - f * sat);
    const t = val * (1 - (1 - f) * sat);
    const mod = i % 6;
    if (mod === 0) return [val, t, p];
    if (mod === 1) return [q, val, p];
    if (mod === 2) return [p, val, t];
    if (mod === 3) return [p, q, val];
    if (mod === 4) return [t, p, val];
    return [val, p, q];
  }

  return [clamp01(x), clamp01(y), clamp01(z)];
}

function sendControlUpdate() {
  if (ws.readyState !== WebSocket.OPEN) {
    return;
  }

  ws.send(
    JSON.stringify({
      type: "update_params",
      dt: parseFloat(dtSlider.value),
      speed: parseFloat(speedSlider.value),
      send_every: parseInt(sendEverySlider.value, 10),
      point_size: parseFloat(pointSizeSlider.value),
      color_scheme: currentColorScheme,
    }),
  );
}

function updateControlValues() {
  dtValue.textContent = Number(dtSlider.value).toFixed(2);
  speedValue.textContent = Number(speedSlider.value).toFixed(1);
  sendEveryValue.textContent = sendEverySlider.value;
  pointSizeValue.textContent = Number(pointSizeSlider.value).toFixed(1);
}

function updateStatsLabels() {
  const now = performance.now();
  const realtimeT = (now - wsStartTime) / 1000;
  realtimeValue.textContent = `${realtimeT.toFixed(1)}s`;
  physicsValue.textContent = `${physicsTime.toFixed(1)}s`;
  fpsValue.textContent = latestFps.toFixed(1);
}

function updatePointSize(newSize) {
  pointSize = Number(newSize);
  gl.useProgram(program);
  gl.uniform1f(uPointSize, pointSize);
}

updateControlValues();

dtSlider.oninput = () => {
  updateControlValues();
  sendControlUpdate();
};

speedSlider.oninput = () => {
  updateControlValues();
  sendControlUpdate();
};

sendEverySlider.oninput = () => {
  updateControlValues();
  sendControlUpdate();
};

pointSizeSlider.oninput = () => {
  updateControlValues();
  updatePointSize(pointSizeSlider.value);
  sendControlUpdate();
};

colorSchemeSelect.onchange = () => {
  currentColorScheme = colorSchemeSelect.value;
  sendControlUpdate();
};

document.getElementById("randomizeSeedBtn").addEventListener("click", () => {
  seedInput.value = String(Math.floor(Math.random() * 1e9));
});

document.getElementById("applySeedBtn").addEventListener("click", () => {
  const seed = parseInt(seedInput.value, 10);
  if (!Number.isFinite(seed)) {
    return;
  }
  ws.send(JSON.stringify({ type: "set_seed", seed }));
});

ws.onopen = () => {
  wsStartTime = performance.now();
  fpsWindowStart = wsStartTime;
  fpsFrameCount = 0;
  latestFps = 0;
  sendControlUpdate();
};

const vertSrc = `
attribute vec3 a_pos;
attribute vec3 a_rgb;

uniform float u_box;
uniform float u_pointSize;

varying vec3 v_rgb;

void main() {
    vec3 p = a_pos / u_box;
    vec2 clip = p.xy * 2.0 - 1.0;
    clip.y = -clip.y;

    gl_Position = vec4(clip, 0.0, 1.0);
    gl_PointSize = u_pointSize;

    v_rgb = clamp(a_rgb, 0.0, 1.0);
}
`;

const fragSrc = `
precision mediump float;
varying vec3 v_rgb;

void main() {
    vec2 c = gl_PointCoord - vec2(0.5);
    if(dot(c,c) > 0.25) discard;
    gl_FragColor = vec4(v_rgb, 1.0);
}
`;

function makeShader(type, source) {
  const shader = gl.createShader(type);
  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    throw new Error(gl.getShaderInfoLog(shader));
  }
  return shader;
}

function makeProgram(vs, fs) {
  const program = gl.createProgram();
  gl.attachShader(program, makeShader(gl.VERTEX_SHADER, vs));
  gl.attachShader(program, makeShader(gl.FRAGMENT_SHADER, fs));
  gl.linkProgram(program);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    throw new Error(gl.getProgramInfoLog(program));
  }
  return program;
}

const program = makeProgram(vertSrc, fragSrc);
gl.useProgram(program);

const aPos = gl.getAttribLocation(program, "a_pos");
const aRgb = gl.getAttribLocation(program, "a_rgb");
const uBox = gl.getUniformLocation(program, "u_box");
const uPointSize = gl.getUniformLocation(program, "u_pointSize");

const buffer = gl.createBuffer();
let drawCount = 0;
let stride = 0;
let boxSize = 1;
let pointSize = 3;
let running = true;

updatePointSize(pointSizeSlider.value);

function resizeCanvas() {
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.floor(canvas.clientWidth * dpr);
  canvas.height = Math.floor(canvas.clientHeight * dpr);
  gl.viewport(0, 0, canvas.width, canvas.height);
}

window.addEventListener("resize", resizeCanvas);
resizeCanvas();

gl.clearColor(0.06, 0.07, 0.1, 1.0);

function render() {
  gl.clear(gl.COLOR_BUFFER_BIT);
  if (drawCount > 0 && stride > 0) {
    gl.useProgram(program);
    gl.bindBuffer(gl.ARRAY_BUFFER, buffer);

    const strideBytes = stride * 4;
    gl.enableVertexAttribArray(aPos);
    gl.vertexAttribPointer(aPos, 3, gl.FLOAT, false, strideBytes, 0);

    const colorOffsetBytes = 3 * 4;
    gl.enableVertexAttribArray(aRgb);
    gl.vertexAttribPointer(aRgb, 3, gl.FLOAT, false, strideBytes, colorOffsetBytes);

    gl.uniform1f(uBox, boxSize);
    gl.uniform1f(uPointSize, pointSize);

    gl.drawArrays(gl.POINTS, 0, drawCount);
  }

  updateStatsLabels();
  requestAnimationFrame(render);
}
requestAnimationFrame(render);

ws.onmessage = (event) => {
  if (typeof event.data === "string") {
    const msg = JSON.parse(event.data);
    if (msg.type === "params") {
      dtSlider.value = String(msg.dt);
      speedSlider.value = String(msg.speed);
      sendEverySlider.value = String(msg.send_every);
      if (typeof msg.point_size === "number") {
        pointSizeSlider.value = String(msg.point_size);
        updatePointSize(msg.point_size);
      }
      if (typeof msg.physics_t === "number") {
        physicsTime = msg.physics_t;
      }
      if (typeof msg.seed === "number") {
        seedInput.value = String(msg.seed);
      }
      if (typeof msg.color_scheme === "string") {
        currentColorScheme = msg.color_scheme;
        colorSchemeSelect.value = msg.color_scheme;
      }
      updateControlValues();
    } else if (msg.type === "stats" && typeof msg.physics_t === "number") {
      physicsTime = msg.physics_t;
    } else if (msg.type === "seed" && typeof msg.seed === "number") {
      seedInput.value = String(msg.seed);
    }
    return;
  }

  const frameBuffer = event.data;
  const dv = new DataView(frameBuffer);

  const posDim = dv.getInt32(0, true);
  const stateDim = dv.getInt32(4, true);
  const N = dv.getInt32(8, true);
  boxSize = dv.getFloat32(12, true);

  const arr = new Float32Array(frameBuffer, 16);
  const inputStride = posDim + stateDim;
  stride = 6;

  if (stateDim < 3 || posDim < 3 || arr.length < N * inputStride) {
    return;
  }

  const interleaved = new Float32Array(N * stride);
  for (let i = 0; i < N; i += 1) {
    const srcBase = i * inputStride;
    const dstBase = i * stride;

    interleaved[dstBase + 0] = arr[srcBase + 0];
    interleaved[dstBase + 1] = arr[srcBase + 1];
    interleaved[dstBase + 2] = arr[srcBase + 2];

    const rgb = stateToRgb(
      [arr[srcBase + posDim + 0], arr[srcBase + posDim + 1], arr[srcBase + posDim + 2]],
      currentColorScheme,
    );
    interleaved[dstBase + 3] = rgb[0];
    interleaved[dstBase + 4] = rgb[1];
    interleaved[dstBase + 5] = rgb[2];
  }

  gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
  gl.bufferData(gl.ARRAY_BUFFER, interleaved, gl.DYNAMIC_DRAW);
  drawCount = N;

  const now = performance.now();
  fpsFrameCount += 1;
  const elapsedMs = now - fpsWindowStart;
  if (elapsedMs >= 1000) {
    latestFps = (fpsFrameCount * 1000) / elapsedMs;
    fpsFrameCount = 0;
    fpsWindowStart = now;
  }
};

document.getElementById("toggleBtn").addEventListener("click", () => {
  running = !running;
  document.getElementById("toggleBtn").textContent = running ? "Pause" : "Run";
  ws.send(JSON.stringify({ type: "set_running", running }));
});

document.getElementById("resetBtn").addEventListener("click", () => {
  ws.send(JSON.stringify({ type: "reset" }));
});
