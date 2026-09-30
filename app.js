import { FilesetResolver, HandLandmarker } from 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.32/+esm';

const modelUrl = 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task';
const wasmUrl = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.32/wasm';
const colours = [
  ['Lilac', '#a78bfa'], ['Pink', '#ff91bb'], ['Coral', '#ff8578'], ['Amber', '#ffcb73'],
  ['Mint', '#6ce5bb'], ['Sky', '#73caff'], ['Blue', '#7593ff'], ['White', '#ffffff']
];
const $ = (id) => document.getElementById(id);
const video = $('camera');
const canvas = $('drawing');
const ctx = canvas.getContext('2d');
const inferenceCanvas = document.createElement('canvas');
const inferenceCtx = inferenceCanvas.getContext('2d', { willReadFrequently: true });
const cursor = $('cursor');
const palette = $('palette');
const hint = $('hint');
let stream;
let landmarker;
let frameId;
let lastVideoTime = -1;
let lastPoint = null;
let lastMode = null;
let ink = colours[0][1];
let hoveredColour = null;
let hoverSince = 0;
let canvasWidth = 0;
let canvasHeight = 0;

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
  lastPoint = null;
}
window.addEventListener('resize', resizeCanvas);
resizeCanvas();

function selectColour(colour) {
  ink = colour;
  $('current-ink').style.background = colour;
  $('current-ink').style.boxShadow = `0 0 14px ${colour}88`;
  document.querySelectorAll('.swatch').forEach((element) => {
    element.classList.toggle('selected', element.dataset.colour === colour);
  });
}

for (const [name, colour] of colours) {
  const button = document.createElement('button');
  button.type = 'button';
  button.className = 'swatch';
  button.style.setProperty('--swatch', colour);
  button.dataset.colour = colour;
  button.setAttribute('aria-label', name);
  button.title = name;
  button.addEventListener('click', () => selectColour(colour));
  $('swatches').append(button);
}
selectColour(ink);

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
  lastPoint = null;
  lastMode = null;
  cursor.style.display = 'none';
  palette.hidden = true;
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

function isOpen(hand) {
  const wrist = hand[0];
  const palm = hand[9];
  const scale = Math.hypot(palm.x - wrist.x, palm.y - wrist.y) || 0.01;
  return [8, 12, 16, 20].filter((tip) =>
    Math.hypot(hand[tip].x - wrist.x, hand[tip].y - wrist.y) >
    Math.hypot(hand[tip - 2].x - wrist.x, hand[tip - 2].y - wrist.y) + scale * 0.22
  ).length >= 3;
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

function updatePalette(point) {
  const swatches = [...document.querySelectorAll('.swatch')];
  const hit = swatches.find((element) => {
    const rect = element.getBoundingClientRect();
    return point.x >= rect.left - 7 && point.x <= rect.right + 7 && point.y >= rect.top - 7 && point.y <= rect.bottom + 7;
  });
  swatches.forEach((element) => element.classList.toggle('hovered', element === hit));
  if (hit !== hoveredColour) {
    hoveredColour = hit;
    hoverSince = performance.now();
  }
  if (hit && performance.now() - hoverSince > 350) selectColour(hit.dataset.colour);
}

function handleHands(result) {
  let right;
  let left;
  result.landmarks.forEach((hand, index) => {
    const label = result.handedness[index]?.[0]?.categoryName;
    if (label === 'Right') right = hand;
    if (label === 'Left') left = hand;
  });
  const singleHand = result.landmarks.length === 1;
  const pointer = singleHand ? result.landmarks[0] : right;
  const drawing = singleHand ? thumbExtended(pointer) : !!left && isOpen(left);
  const mode = singleHand ? 'one' : 'two';
  palette.hidden = !pointer || (!singleHand && !left) || drawing;
  if (!pointer) {
    cursor.style.display = 'none';
    lastPoint = null;
    lastMode = null;
    setHint(result.landmarks.length ? 'Show your right hand to move the cursor' : 'Show a hand to begin');
    return;
  }
  if (mode !== lastMode) lastPoint = null;
  lastMode = mode;
  const point = screenPoint(pointer[8]);
  cursor.style.display = 'grid';
  cursor.style.left = `${point.x}px`;
  cursor.style.top = `${point.y}px`;
  cursor.classList.toggle('drawing', drawing);
  if (!singleHand && !left) {
    lastPoint = null;
    setHint('Show your left hand to draw or pick a colour');
  } else if (drawing) {
    setHint(singleHand ? 'Drawing · tuck your thumb in to choose a colour' : 'Drawing · close your left hand to choose a colour');
    ctx.strokeStyle = ink;
    ctx.fillStyle = ink;
    ctx.lineWidth = 5;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.shadowColor = ink;
    ctx.shadowBlur = 15;
    if (lastPoint && Math.hypot(point.x - lastPoint.x, point.y - lastPoint.y) < 110) {
      ctx.beginPath();
      ctx.moveTo(lastPoint.x, lastPoint.y);
      ctx.lineTo(point.x, point.y);
      ctx.stroke();
    } else {
      ctx.beginPath();
      ctx.arc(point.x, point.y, 2.5, 0, Math.PI * 2);
      ctx.fill();
    }
    lastPoint = point;
  } else {
    lastPoint = null;
    setHint(singleHand ? 'Tuck thumb in · point at a colour' : 'Point your right index finger at a colour');
    updatePalette(point);
  }
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
$('clear').addEventListener('click', () => ctx.clearRect(0, 0, canvasWidth, canvasHeight));
$('save').addEventListener('click', () => {
  const link = document.createElement('a');
  link.download = 'skywriter.png';
  link.href = canvas.toDataURL('image/png');
  link.click();
});
window.addEventListener('pagehide', () => stream?.getTracks().forEach((track) => track.stop()));
