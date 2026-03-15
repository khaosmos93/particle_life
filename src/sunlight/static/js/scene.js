import * as THREE from "https://cdn.jsdelivr.net/npm/three@0.164.1/build/three.module.js";
import { OrbitControls } from "https://cdn.jsdelivr.net/npm/three@0.164.1/examples/jsm/controls/OrbitControls.js";

const root = document.getElementById("scene-root");

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x030306);

const camera = new THREE.PerspectiveCamera(60, root.clientWidth / root.clientHeight, 0.1, 1000);
camera.position.set(0, 3.5, 10);

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(window.devicePixelRatio);
renderer.setSize(root.clientWidth, root.clientHeight);
root.appendChild(renderer.domElement);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.target.set(0, 0, 0);

const ambient = new THREE.AmbientLight(0xffffff, 0.2);
scene.add(ambient);

const sunGeometry = new THREE.SphereGeometry(1.5, 48, 48);
const sunMaterial = new THREE.MeshBasicMaterial({ color: 0xffd65c });
const sun = new THREE.Mesh(sunGeometry, sunMaterial);
sun.position.set(-6, 0, 0);
scene.add(sun);

const sunLight = new THREE.PointLight(0xfff1b8, 2.2, 60);
sunLight.position.copy(sun.position);
scene.add(sunLight);

const earthGeometry = new THREE.SphereGeometry(1, 32, 32);
const earthMaterial = new THREE.MeshBasicMaterial({
  color: 0x000000,
  wireframe: true,
});
const earth = new THREE.Mesh(earthGeometry, earthMaterial);
earth.position.set(2, 0, 0);
scene.add(earth);

const axisHelper = new THREE.AxesHelper(2.5);
scene.add(axisHelper);

function onResize() {
  const { clientWidth, clientHeight } = root;
  camera.aspect = clientWidth / clientHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(clientWidth, clientHeight);
}

window.addEventListener("resize", onResize);

function animate() {
  requestAnimationFrame(animate);
  controls.update();
  renderer.render(scene, camera);
}

animate();
