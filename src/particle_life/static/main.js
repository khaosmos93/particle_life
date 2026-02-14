const canvas = document.getElementById("canvas");
const gl = canvas.getContext("webgl");

if (!gl) {
  throw new Error("WebGL unavailable");
}

const vertexSrc = `
attribute vec2 a_pos;
attribute float a_species;
uniform vec2 u_scale;
varying vec3 v_color;

void main() {
  vec2 clip = a_pos * 2.0 - 1.0;
  gl_Position = vec4(clip.x, -clip.y, 0.0, 1.0);
  gl_PointSize = 3.0 * u_scale.x;

  int idx = int(mod(a_species, 5.0));
  if (idx == 0) v_color = vec3(1.0, 0.45, 0.35);
  else if (idx == 1) v_color = vec3(0.4, 0.85, 1.0);
  else if (idx == 2) v_color = vec3(0.62, 1.0, 0.45);
  else if (idx == 3) v_color = vec3(1.0, 0.95, 0.45);
  else v_color = vec3(0.88, 0.52, 1.0);
}
`;

const fragmentSrc = `
precision mediump float;
varying vec3 v_color;

void main() {
  vec2 c = gl_PointCoord - vec2(0.5);
  float d = dot(c, c);
  if (d > 0.25) discard;
  gl_FragColor = vec4(v_color, 0.95);
}
`;

function compile(type, src) {
  const shader = gl.createShader(type);
  gl.shaderSource(shader, src);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    throw new Error(gl.getShaderInfoLog(shader));
  }
  return shader;
}

const program = gl.createProgram();
gl.attachShader(program, compile(gl.VERTEX_SHADER, vertexSrc));
gl.attachShader(program, compile(gl.FRAGMENT_SHADER, fragmentSrc));
gl.linkProgram(program);
if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
  throw new Error(gl.getProgramInfoLog(program));
}

const posBuffer = gl.createBuffer();
const speciesBuffer = gl.createBuffer();

const aPos = gl.getAttribLocation(program, "a_pos");
const aSpecies = gl.getAttribLocation(program, "a_species");
const uScale = gl.getUniformLocation(program, "u_scale");

let pointCount = 0;
let positions = new Float32Array();
let species = new Float32Array();

function resize() {
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.floor(window.innerWidth * dpr);
  canvas.height = Math.floor(window.innerHeight * dpr);
  gl.viewport(0, 0, canvas.width, canvas.height);
}

window.addEventListener("resize", resize);
resize();

function draw() {
  gl.clearColor(0.02, 0.03, 0.08, 1.0);
  gl.clear(gl.COLOR_BUFFER_BIT);

  if (!pointCount) return;

  gl.useProgram(program);
  gl.uniform2f(uScale, window.devicePixelRatio || 1, 1);

  gl.bindBuffer(gl.ARRAY_BUFFER, posBuffer);
  gl.bufferData(gl.ARRAY_BUFFER, positions, gl.DYNAMIC_DRAW);
  gl.enableVertexAttribArray(aPos);
  gl.vertexAttribPointer(aPos, 2, gl.FLOAT, false, 0, 0);

  gl.bindBuffer(gl.ARRAY_BUFFER, speciesBuffer);
  gl.bufferData(gl.ARRAY_BUFFER, species, gl.DYNAMIC_DRAW);
  gl.enableVertexAttribArray(aSpecies);
  gl.vertexAttribPointer(aSpecies, 1, gl.FLOAT, false, 0, 0);

  gl.drawArrays(gl.POINTS, 0, pointCount);
}

function consumeFrame(buffer) {
  const data = new Float32Array(buffer);
  pointCount = Math.floor(data.length / 3);
  positions = new Float32Array(pointCount * 2);
  species = new Float32Array(pointCount);

  for (let i = 0; i < pointCount; i += 1) {
    positions[i * 2] = data[i * 3];
    positions[i * 2 + 1] = data[i * 3 + 1];
    species[i] = data[i * 3 + 2];
  }

  draw();
}

const wsScheme = window.location.protocol === "https:" ? "wss" : "ws";
const ws = new WebSocket(`${wsScheme}://${window.location.host}/ws`);
ws.binaryType = "arraybuffer";
ws.onmessage = (event) => consumeFrame(event.data);
