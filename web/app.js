const canvas = document.getElementById("glcanvas");
const gl = canvas.getContext("webgl");
if (!gl) throw new Error("WebGL not supported");

const ws = new WebSocket(`${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws`);
ws.binaryType = "arraybuffer";

const ids = ["dtSlider","speedSlider","sendEverySlider","pointSize","seedInput","colorSchemeSelect","pbcTilingToggle","dtValue","speedValue","sendEveryValue","pointSizeValue","fpsValue","realtimeValue","physicsValue","toggleBtn","resetBtn","randomizeSeedBtn","applySeedBtn"];
const el = Object.fromEntries(ids.map((id) => [id, document.getElementById(id)]));

let pos = new Float32Array(0), rgb = new Float32Array(0), count = 0, box = 1, pointSize = 15, running = true, showPbc = false;
let start = performance.now(), fCount = 0, fps = 0, lastFpsT = performance.now(), physicsT = 0;

function clamp01(v) { return Math.max(0, Math.min(1, v)); }
function stateToRgb(x, y, z, mode) {
  if (mode === "normalize") {
    const n = Math.hypot(x, y, z) || 1; return [clamp01(0.5 + 0.5 * x / n), clamp01(0.5 + 0.5 * y / n), clamp01(0.5 + 0.5 * z / n)];
  }
  if (mode === "abs") {
    const s = Math.abs(x)+Math.abs(y)+Math.abs(z) || 1; return [Math.abs(x)/s, Math.abs(y)/s, Math.abs(z)/s];
  }
  if (mode === "softmax") {
    const ex=Math.exp(x), ey=Math.exp(y), ez=Math.exp(z), d=ex+ey+ez||1; return [ex/d, ey/d, ez/d];
  }
  if (mode === "hsv_like") {
    const h=(Math.atan2(y,x)/(2*Math.PI)+1)%1; const i=Math.floor(h*6), f=h*6-i; const p=z*(1-1), q=z*(1-f), t=z*f;
    if(i%6===0) return [z,t,p]; if(i%6===1) return [q,z,p]; if(i%6===2) return [p,z,t]; if(i%6===3) return [p,q,z]; if(i%6===4) return [t,p,z]; return [z,p,q];
  }
  return [clamp01(x), clamp01(y), clamp01(z)];
}

function sendParams() {
  if (ws.readyState !== WebSocket.OPEN) return;
  ws.send(JSON.stringify({type:"update_params", dt:+el.dtSlider.value, speed:+el.speedSlider.value, send_every:+el.sendEverySlider.value, point_size:+el.pointSize.value, color_scheme:el.colorSchemeSelect.value}));
}

function updateLabels() {
  el.dtValue.textContent = (+el.dtSlider.value).toFixed(3);
  el.speedValue.textContent = (+el.speedSlider.value).toFixed(1);
  el.sendEveryValue.textContent = el.sendEverySlider.value;
  el.pointSizeValue.textContent = (+el.pointSize.value).toFixed(1);
}
["dtSlider","speedSlider","sendEverySlider","pointSize"].forEach((id)=>el[id].oninput=()=>{updateLabels(); pointSize=+el.pointSize.value; sendParams();});
el.colorSchemeSelect.onchange = sendParams;
el.pbcTilingToggle.onchange = ()=>{showPbc = el.pbcTilingToggle.checked;};
el.toggleBtn.onclick = ()=>{running=!running; el.toggleBtn.textContent=running?"Pause":"Resume"; ws.send(JSON.stringify({type:"set_running", running}));};
el.resetBtn.onclick = ()=>ws.send(JSON.stringify({type:"reset"}));
el.randomizeSeedBtn.onclick = ()=>{el.seedInput.value = String(Math.floor(Math.random()*1e9));};
el.applySeedBtn.onclick = ()=>ws.send(JSON.stringify({type:"set_seed", seed:+el.seedInput.value}));
updateLabels();

const vert=`attribute vec3 a_pos; attribute vec3 a_rgb; uniform float u_box; uniform float u_size; uniform vec2 u_off; varying vec3 v; void main(){ vec2 clip=((a_pos.xy/u_box)+u_off)*2.0-1.0; clip.y=-clip.y; gl_Position=vec4(clip,0.,1.); gl_PointSize=u_size; v=a_rgb; }`;
const frag=`precision mediump float; varying vec3 v; void main(){ vec2 c=gl_PointCoord-vec2(.5); if(dot(c,c)>.25) discard; gl_FragColor=vec4(v,1.); }`;
function sh(t,s){const x=gl.createShader(t);gl.shaderSource(x,s);gl.compileShader(x);return x;}
const prog=gl.createProgram(); gl.attachShader(prog, sh(gl.VERTEX_SHADER, vert)); gl.attachShader(prog, sh(gl.FRAGMENT_SHADER, frag)); gl.linkProgram(prog); gl.useProgram(prog);
const posB=gl.createBuffer(), rgbB=gl.createBuffer();
const aPos=gl.getAttribLocation(prog,"a_pos"), aRgb=gl.getAttribLocation(prog,"a_rgb");
const uBox=gl.getUniformLocation(prog,"u_box"), uSize=gl.getUniformLocation(prog,"u_size"), uOff=gl.getUniformLocation(prog,"u_off");

gl.enableVertexAttribArray(aPos); gl.enableVertexAttribArray(aRgb); gl.clearColor(0.06,0.07,0.1,1);

function resize(){canvas.width=innerWidth; canvas.height=innerHeight; gl.viewport(0,0,canvas.width,canvas.height);} addEventListener("resize",resize); resize();

ws.onmessage = (ev) => {
  if (typeof ev.data === "string") {
    const msg = JSON.parse(ev.data);
    if (msg.type === "params") {
      if (msg.dt) el.dtSlider.value = msg.dt;
      if (msg.speed) el.speedSlider.value = msg.speed;
      if (msg.send_every) el.sendEverySlider.value = msg.send_every;
      if (msg.point_size) el.pointSize.value = msg.point_size;
      if (msg.seed !== undefined) el.seedInput.value = msg.seed;
      if (msg.color_scheme) el.colorSchemeSelect.value = msg.color_scheme;
      updateLabels();
    }
    if (msg.type === "perf") physicsT = msg.physics_t || physicsT;
    return;
  }
  const b = ev.data;
  const hi = new Int32Array(b, 0, 3); box = new Float32Array(b, 12, 1)[0];
  const row = hi[0] + hi[1]; count = hi[2];
  const body = new Float32Array(b, 16, count * row);
  pos = new Float32Array(count * 3); rgb = new Float32Array(count * 3);
  for (let i=0;i<count;i++) {
    pos[i*3] = body[i*row]; pos[i*3+1] = body[i*row+1]; pos[i*3+2] = body[i*row+2] || 0;
    const [r,g,bb]=stateToRgb(body[i*row+hi[0]]||0, body[i*row+hi[0]+1]||0, body[i*row+hi[0]+2]||0, el.colorSchemeSelect.value);
    rgb[i*3]=r; rgb[i*3+1]=g; rgb[i*3+2]=bb;
  }
  fCount++;
};

function draw(offx, offy){
  gl.uniform2f(uOff, offx, offy); gl.uniform1f(uBox, box); gl.uniform1f(uSize, pointSize);
  gl.bindBuffer(gl.ARRAY_BUFFER, posB); gl.bufferData(gl.ARRAY_BUFFER, pos, gl.DYNAMIC_DRAW); gl.vertexAttribPointer(aPos, 3, gl.FLOAT, false, 0, 0);
  gl.bindBuffer(gl.ARRAY_BUFFER, rgbB); gl.bufferData(gl.ARRAY_BUFFER, rgb, gl.DYNAMIC_DRAW); gl.vertexAttribPointer(aRgb, 3, gl.FLOAT, false, 0, 0);
  gl.drawArrays(gl.POINTS, 0, count);
}

function loop() {
  gl.clear(gl.COLOR_BUFFER_BIT);
  if (count) {
    if (showPbc) for (let x=-1;x<=1;x++) for (let y=-1;y<=1;y++) draw(x,y); else draw(0,0);
  }
  const now = performance.now();
  if (now - lastFpsT > 1000) { fps = fCount * 1000 / (now - lastFpsT); fCount = 0; lastFpsT = now; }
  el.fpsValue.textContent = fps.toFixed(1);
  el.realtimeValue.textContent = `${((now-start)/1000).toFixed(1)}s`;
  el.physicsValue.textContent = `${physicsT.toFixed(1)}s`;
  requestAnimationFrame(loop);
}
requestAnimationFrame(loop);
