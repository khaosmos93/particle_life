const canvas = document.getElementById("glcanvas");
const gl = canvas.getContext("webgl");

if (!gl) {
  throw new Error("WebGL not supported");
}

const protocol = location.protocol === "https:" ? "wss" : "ws";
const ws = new WebSocket(`${protocol}://${location.host}/ws`);
ws.binaryType = "arraybuffer";

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

    v_rgb = clamp(a_rgb * 0.5 + 0.5, 0.0, 1.0);
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
let currentPosDim = 3;

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

    const colorOffsetBytes = currentPosDim * 4;
    gl.enableVertexAttribArray(aRgb);
    gl.vertexAttribPointer(aRgb, 3, gl.FLOAT, false, strideBytes, colorOffsetBytes);

    gl.uniform1f(uBox, boxSize);
    gl.uniform1f(uPointSize, pointSize);

    gl.drawArrays(gl.POINTS, 0, drawCount);
  }

  requestAnimationFrame(render);
}
requestAnimationFrame(render);

ws.onmessage = (event) => {
  const frameBuffer = event.data;
  const dv = new DataView(frameBuffer);

  const posDim = dv.getInt32(0, true);
  const stateDim = dv.getInt32(4, true);
  const N = dv.getInt32(8, true);
  boxSize = dv.getFloat32(12, true);

  const arr = new Float32Array(frameBuffer, 16);
  stride = posDim + stateDim;
  currentPosDim = posDim;

  if (stateDim < 3 || posDim < 3 || arr.length < N * stride) {
    return;
  }

  gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
  gl.bufferData(gl.ARRAY_BUFFER, arr, gl.DYNAMIC_DRAW);
  drawCount = N;
};

document.getElementById("toggleBtn").addEventListener("click", () => {
  running = !running;
  document.getElementById("toggleBtn").textContent = running ? "Pause" : "Run";
  ws.send(JSON.stringify({ type: "set_running", running }));
});

document.getElementById("resetBtn").addEventListener("click", () => {
  ws.send(JSON.stringify({ type: "reset" }));
});

document.getElementById("pointSize").addEventListener("input", (e) => {
  pointSize = Number(e.target.value);
});
