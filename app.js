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
const fadeLayer = document.createElement('canvas');
const fadeCtx = fadeLayer.getContext('2d');
const inferenceCanvas = document.createElement('canvas');
const inferenceCtx = inferenceCanvas.getContext('2d', { willReadFrequently: true });
const markers = $('markers');
const hint = $('hint');
let stream;
let landmarker;
let frameId;
let lastVideoTime = -1;
const lastPoints = new Map();
const lastDrawnPoints = new Map();
const fingerMarkers = new Map();
let canvasWidth = 0;
let canvasHeight = 0;
let fadeFrame;
let lastRenderTime = 0;
let fadingStrokes = [];
const fadeButton = $('fade');
const fadeDuration = 10000;
const fadeInterval = 500;
const fadeOpacity = (age) => Math.max(0, 1 - age / fadeDuration) ** 2;

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
  fadeLayer.width = canvas.width;
  fadeLayer.height = canvas.height;
  fadeCtx.setTransform(dpr, 0, 0, dpr, 0, 0);
  canvas.style.width = `${width}px`;
  canvas.style.height = `${height}px`;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.shadowBlur = 0;
  ctx.shadowColor = 'transparent';
  if (backup.width && backup.height && fadeButton?.getAttribute('aria-pressed') !== 'true') {
    ctx.drawImage(backup, 0, 0, canvasWidth, canvasHeight);
  }
  canvasWidth = width;
  canvasHeight = height;
  if (fadeButton?.getAttribute('aria-pressed') === 'true') {
    for (const stroke of fadingStrokes) {
      if (stroke.image) {
        const resized = document.createElement('canvas');
        resized.width = canvas.width;
        resized.height = canvas.height;
        resized.getContext('2d').drawImage(stroke.image, 0, 0, resized.width, resized.height);
        stroke.image = resized;
      } else {
        stroke.point = { x: stroke.point.x * width / canvasWidth, y: stroke.point.y * height / canvasHeight };
        if (stroke.from) stroke.from = { x: stroke.from.x * width / canvasWidth, y: stroke.from.y * height / canvasHeight };
      }
    }
  }
  lastPoints.clear();
  lastDrawnPoints.clear();
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
  lastDrawnPoints.clear();
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

function drawFinger(point, previous, colour, key) {
  if (fadeButton.getAttribute('aria-pressed') === 'true') {
    const lastDrawn = previous ? lastDrawnPoints.get(key) : null;
    if (lastDrawn && Math.hypot(point.x - lastDrawn.x, point.y - lastDrawn.y) < 2) return;
    const from = lastDrawn && Math.hypot(point.x - lastDrawn.x, point.y - lastDrawn.y) < 110 ? lastDrawn : null;
    fadingStrokes.push({ point, from, colour, born: Math.floor(performance.now() / fadeInterval) * fadeInterval });
    lastDrawnPoints.set(key, point);
    return;
  }
  lastDrawnPoints.delete(key);
  const from = previous && Math.hypot(point.x - previous.x, point.y - previous.y) < 110 ? previous : null;
  paintStroke(point, from, colour);
}

function paintStroke(point, from, colour, glow = true, target = ctx) {
  target.strokeStyle = colour;
  target.fillStyle = colour;
  target.lineWidth = 5;
  target.lineCap = 'round';
  target.lineJoin = 'round';
  target.shadowColor = glow ? colour : 'transparent';
  target.shadowBlur = glow ? 15 : 0;
  target.beginPath();
  if (from) {
    target.moveTo(from.x, from.y);
    target.lineTo(point.x, point.y);
    target.stroke();
  } else {
    target.arc(point.x, point.y, 2.5, 0, Math.PI * 2);
    target.fill();
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
      drawFinger(point, lastPoints.get(key), colour, key);
      lastPoints.set(key, point);
    });
  });
  fingerMarkers.forEach((marker, key) => {
    if (!active.has(key)) {
      marker.style.display = 'none';
      lastPoints.delete(key);
      lastDrawnPoints.delete(key);
    }
  });
  setHint(result.landmarks.length ? 'Extended fingers draw · curl a finger to lift its brush' : 'Show a hand to begin');
}

function fadeDrawing(now) {
  if (now - lastRenderTime < 30) {
    fadeFrame = requestAnimationFrame(fadeDrawing);
    return;
  }
  lastRenderTime = now;
  ctx.globalAlpha = 1;
  ctx.shadowBlur = 0;
  ctx.shadowColor = 'transparent';
  ctx.clearRect(0, 0, canvasWidth, canvasHeight);
  fadingStrokes = fadingStrokes.filter((stroke) => now - stroke.born < fadeDuration);
  const layers = new Map();
  for (const stroke of fadingStrokes) {
    if (stroke.image) {
      ctx.globalAlpha = fadeOpacity(now - stroke.born);
      ctx.shadowBlur = 0;
      ctx.shadowColor = 'transparent';
      ctx.drawImage(stroke.image, 0, 0, canvasWidth, canvasHeight);
    } else {
      const bucket = stroke.born;
      if (!layers.has(bucket)) layers.set(bucket, []);
      layers.get(bucket).push(stroke);
    }
  }
  for (const [bucket, strokes] of layers) {
    fadeCtx.clearRect(0, 0, canvasWidth, canvasHeight);
    for (const stroke of strokes) paintStroke(stroke.point, stroke.from, stroke.colour, false, fadeCtx);
    ctx.globalAlpha = fadeOpacity(now - bucket);
    ctx.drawImage(fadeLayer, 0, 0, canvasWidth, canvasHeight);
  }
  ctx.globalAlpha = 1;
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
  lastDrawnPoints.clear();
  if (enabled) {
    const image = document.createElement('canvas');
    image.width = canvas.width;
    image.height = canvas.height;
    image.getContext('2d').drawImage(canvas, 0, 0);
    fadingStrokes = [{ image, born: performance.now() }];
    fadeFrame = requestAnimationFrame(fadeDrawing);
  } else {
    cancelAnimationFrame(fadeFrame);
    fadingStrokes = [];
  }
});
$('clear').addEventListener('click', () => {
  fadingStrokes = [];
  ctx.clearRect(0, 0, canvasWidth, canvasHeight);
});
$('save').addEventListener('click', () => {
  const link = document.createElement('a');
  link.download = 'skywriter.png';
  link.href = canvas.toDataURL('image/png');
  link.click();
});
window.addEventListener('pagehide', () => stream?.getTracks().forEach((track) => track.stop()));
