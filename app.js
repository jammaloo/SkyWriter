import { FilesetResolver, HandLandmarker } from 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.32/+esm';

const modelUrl = 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task';
const wasmUrl = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.32/wasm';
const fingers = [
  { name: 'Thumb', tip: 4, colour: '#a78bfa' },
  { name: 'Index', tip: 8, colour: '#ff91bb' },
  { name: 'Middle', tip: 12, colour: '#ffcb73' },
  { name: 'Ring', tip: 16, colour: '#6ce5bb' },
  { name: 'Pinky', tip: 20, colour: '#73caff' }
];
const handColours = {
  Left: fingers.map(({ colour }) => colour),
  Right: ['#d7a5ff', '#ff8578', '#ffe796', '#8ca5ff', '#ffffff']
};
const $ = (id) => document.getElementById(id);
const video = $('camera');
const canvas = $('drawing');
const ctx = canvas.getContext('2d');
const inferenceCanvas = document.createElement('canvas');
const inferenceCtx = inferenceCanvas.getContext('2d', { willReadFrequently: true });
const markers = $('markers');
const hint = $('hint');
let stream;
let landmarker;
let frameId;
let lastVideoTime = -1;
const lastPoints = new Map();
const fingerMarkers = new Map();
let canvasWidth = 0;
let canvasHeight = 0;
let fadeFrame;
let lastFadeTime;
const fadeButton = $('fade');

function resizeCanvas() {
  const width = window.innerWidth;
  const height = window.innerHeight;
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const backup = document.createElement('canvas');
  backup.width = canvas.width;
  backup.height = canvas.height;
  if (backup.width && backup.height) backup.getContext('2d').drawImage(canvas, 0, 0);
  canvas.width = Math.round(width * dpr);
  canvas.height = Math.round(height * dpr);
  canvas.style.width = `${width}px`;
  canvas.style.height = `${height}px`;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  if (backup.width && backup.height) ctx.drawImage(backup, 0, 0, canvasWidth, canvasHeight);
  canvasWidth = width;
  canvasHeight = height;
  lastPoints.clear();
}
window.addEventListener('resize', resizeCanvas);
resizeCanvas();

for (const side of ['Left', 'Right']) {
  fingers.forEach(({ name }, index) => {
    const key = `${side}-${name}`;
    const marker = document.createElement('div');
    marker.className = 'finger-marker';
    marker.style.setProperty('--finger-colour', handColours[side][index]);
    marker.setAttribute('aria-hidden', 'true');
    markers.append(marker);
    fingerMarkers.set(key, marker);
  });
}

function setStatus(message) {
  $('status-text').textContent = message;
}

function setHint(message) {
  if (hint.textContent !== message) hint.textContent = message;
}

function stopCamera() {
  cancelAnimationFrame(frameId);
  stream?.getTracks().forEach((track) => track.stop());
  stream = undefined;
  video.srcObject = null;
  landmarker?.close();
  landmarker = undefined;
  lastVideoTime = -1;
  lastPoints.clear();
  fingerMarkers.forEach((marker) => { marker.style.display = 'none'; });
  $('workspace').hidden = true;
  $('welcome').hidden = false;
  $('camera-toggle').hidden = true;
  document.querySelector('.app').classList.remove('active');
  $('start').disabled = false;
  $('start').textContent = 'Enable camera ↗';
  setStatus('Camera off');
}

async function startCamera() {
  $('error').hidden = true;
  $('start').disabled = true;
  $('start').textContent = 'Starting camera…';
  setStatus('Starting…');
  try {
    if (!navigator.mediaDevices?.getUserMedia) throw new Error('Camera access requires HTTPS or localhost and a supported browser.');
    stream = await navigator.mediaDevices.getUserMedia({ audio: false, video: { facingMode: 'user', width: { ideal: 1280 }, height: { ideal: 720 } } });
    video.srcObject = stream;
    await video.play();
    setStatus('Loading hand tracking…');
    const vision = await FilesetResolver.forVisionTasks(wasmUrl);
    landmarker = await HandLandmarker.createFromOptions(vision, {
      baseOptions: { modelAssetPath: modelUrl, delegate: 'GPU' },
      runningMode: 'VIDEO', numHands: 2,
      minHandDetectionConfidence: 0.6, minHandPresenceConfidence: 0.6, minTrackingConfidence: 0.6
    }).catch(() => HandLandmarker.createFromOptions(vision, {
      baseOptions: { modelAssetPath: modelUrl, delegate: 'CPU' }, runningMode: 'VIDEO', numHands: 2
    }));
    inferenceCanvas.width = Math.min(video.videoWidth, 640);
    inferenceCanvas.height = Math.round(inferenceCanvas.width * video.videoHeight / video.videoWidth);
    $('welcome').hidden = true;
    $('workspace').hidden = false;
    $('camera-toggle').hidden = false;
    document.querySelector('.app').classList.add('active');
    setStatus('Camera live');
    frameId = requestAnimationFrame(trackHands);
  } catch (error) {
    stopCamera();
    const message = error.name === 'NotAllowedError' ? 'Camera permission was denied. Allow camera access in your browser settings and try again.' :
      error.name === 'NotFoundError' ? 'No camera was found. Connect a camera and try again.' :
      error.message || 'Could not start hand tracking. Check your connection and try again.';
    $('error').textContent = message;
    $('error').hidden = false;
  }
}

function fingerExtended(hand, tip) {
  const distance = (a, b) => Math.hypot(hand[a].x - hand[b].x, hand[a].y - hand[b].y);
  const palmLength = distance(0, 9) || 0.01;
  return distance(tip, 0) > distance(tip - 2, 0) + palmLength * 0.22;
}

function thumbExtended(hand) {
  const distance = (a, b) => Math.hypot(hand[a].x - hand[b].x, hand[a].y - hand[b].y);
  const palmWidth = distance(5, 17) || 0.01;
  return distance(4, 5) > palmWidth * 0.8 &&
    distance(4, 17) > distance(3, 17) + palmWidth * 0.25;
}

function screenPoint(landmark) {
  const width = window.innerWidth;
  const height = window.innerHeight;
  const scale = Math.max(width / video.videoWidth, height / video.videoHeight);
  return {
    x: landmark.x * video.videoWidth * scale - (video.videoWidth * scale - width) / 2,
    y: landmark.y * video.videoHeight * scale - (video.videoHeight * scale - height) / 2
  };
}

function drawFinger(point, previous, colour) {
  ctx.strokeStyle = colour;
  ctx.fillStyle = colour;
  ctx.lineWidth = 5;
  ctx.lineCap = 'round';
  ctx.lineJoin = 'round';
  ctx.shadowColor = colour;
  ctx.shadowBlur = 15;
  ctx.beginPath();
  if (previous && Math.hypot(point.x - previous.x, point.y - previous.y) < 110) {
    ctx.moveTo(previous.x, previous.y);
    ctx.lineTo(point.x, point.y);
    ctx.stroke();
  } else {
    ctx.arc(point.x, point.y, 2.5, 0, Math.PI * 2);
    ctx.fill();
  }
}

function handleHands(result) {
  const active = new Set();
  const usedSides = new Set();
  result.landmarks.forEach((hand, index) => {
    const detectedSide = result.handedness[index]?.[0]?.categoryName;
    const side = (detectedSide === 'Left' || detectedSide === 'Right') && !usedSides.has(detectedSide)
      ? detectedSide : usedSides.has('Left') ? 'Right' : 'Left';
    usedSides.add(side);
    fingers.forEach(({ name, tip }, fingerIndex) => {
      const key = `${side}-${name}`;
      if (!(tip === 4 ? thumbExtended(hand) : fingerExtended(hand, tip))) return;
      active.add(key);
      const point = screenPoint(hand[tip]);
      const colour = handColours[side][fingerIndex];
      const marker = fingerMarkers.get(key);
      marker.style.display = 'block';
      marker.style.left = `${point.x}px`;
      marker.style.top = `${point.y}px`;
      drawFinger(point, lastPoints.get(key), colour);
      lastPoints.set(key, point);
    });
  });
  fingerMarkers.forEach((marker, key) => {
    if (!active.has(key)) {
      marker.style.display = 'none';
      lastPoints.delete(key);
    }
  });
  setHint(result.landmarks.length ? 'Extended fingers draw · curl a finger to lift its brush' : 'Show a hand to begin');
}

function fadeDrawing(now) {
  if (lastFadeTime !== undefined) {
    const elapsed = Math.min(now - lastFadeTime, 100);
    ctx.save();
    ctx.globalCompositeOperation = 'destination-out';
    ctx.fillStyle = `rgba(0, 0, 0, ${1 - Math.exp(-elapsed / 5000)})`;
    ctx.fillRect(0, 0, canvasWidth, canvasHeight);
    ctx.restore();
  }
  lastFadeTime = now;
  fadeFrame = requestAnimationFrame(fadeDrawing);
}

function trackHands() {
  if (!stream || !landmarker) return;
  try {
    if (video.readyState >= 2 && video.currentTime !== lastVideoTime) {
      inferenceCtx.save();
      inferenceCtx.translate(inferenceCanvas.width, 0);
      inferenceCtx.scale(-1, 1);
      inferenceCtx.drawImage(video, 0, 0, inferenceCanvas.width, inferenceCanvas.height);
      inferenceCtx.restore();
      handleHands(landmarker.detectForVideo(inferenceCanvas, performance.now()));
      lastVideoTime = video.currentTime;
    }
    frameId = requestAnimationFrame(trackHands);
  } catch (error) {
    stopCamera();
    $('error').textContent = `Hand tracking stopped: ${error.message}`;
    $('error').hidden = false;
  }
}

$('start').addEventListener('click', startCamera);
$('camera-toggle').addEventListener('click', stopCamera);
fadeButton.addEventListener('click', () => {
  const enabled = fadeButton.getAttribute('aria-pressed') !== 'true';
  fadeButton.setAttribute('aria-pressed', String(enabled));
  if (enabled) {
    lastFadeTime = undefined;
    fadeFrame = requestAnimationFrame(fadeDrawing);
  } else {
    cancelAnimationFrame(fadeFrame);
    lastFadeTime = undefined;
  }
});
$('clear').addEventListener('click', () => ctx.clearRect(0, 0, canvasWidth, canvasHeight));
$('save').addEventListener('click', () => {
  const link = document.createElement('a');
  link.download = 'skywriter.png';
  link.href = canvas.toDataURL('image/png');
  link.click();
});
window.addEventListener('pagehide', () => stream?.getTracks().forEach((track) => track.stop()));
