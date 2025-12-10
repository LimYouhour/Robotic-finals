#!/usr/bin/env python3
"""
Gesture-Controlled Robot (ONNX EfficientNet)
--------------------------------------------

Uses a gesture classification ONNX model to drive the AUPPBot.
Detects hand gestures from webcam and maps them to robot movements.
"""

import os
import time
from threading import Thread, Lock
from typing import Tuple, Optional

import cv2
import numpy as np
import onnxruntime as ort
from flask import Flask, Response, jsonify, make_response

try:
    from auppbot import AUPPBot
    AUPPBOT_AVAILABLE = True
except ImportError:
    try:
        from autobot import AUPPBot
        AUPPBOT_AVAILABLE = True
    except ImportError:
        AUPPBOT_AVAILABLE = False
        print("⚠ auppbot/autobot module not found - running in simulation mode only")

# ========== CONFIGURATION ==========
# Path to your gesture ONNX model and class names file
MODEL_PATH = "gesture_model.onnx"
CLASS_NAMES_FILE = "class_names.txt"

# If class_names.txt is missing, fallback to this list (edit as needed)
CLASS_NAMES_FALLBACK = ["01_palm", "02_fist", "03_ok", "04_index-right", "05_index-left"]

# Gesture-to-action mapping (matches class_names.txt)
ACTION_MAP = {
    "01_palm": "stop",
    "02_fist": "backward",
    "03_ok": "forward",
    "04_index-right": "right_then_forward",
    "05_index-left": "left_then_forward",
}

IMG_SIZE = 224  # EfficientNet export uses 224x224

# Camera
CAM_INDEX = 0
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
SKIP_FRAMES = 2
# Set to False to keep the camera feed as-is (no flip)
FLIP_CAMERA = False
JPEG_QUALITY = 80

# Detection thresholds
DEFAULT_CONFIDENCE_THRESHOLD = 0.65
MIN_CONFIDENCE_DIFF = 0.08

# Preprocessing - Gesture model uses ImageNet normalization
USE_IMAGENET_NORM = True

# Robot
ROBOT_PORT = "/dev/ttyUSB0"
DRIVE_SPEED = 15
TURN_SPEED = 23
MOVEMENT_TIME = 2.0  # Move for 2 seconds
TURN_TIME_90 = 1.0   # Time to turn 90 degrees

def load_class_names() -> list[str]:
    if os.path.exists(CLASS_NAMES_FILE):
        with open(CLASS_NAMES_FILE, "r", encoding="utf-8") as f:
            names = [ln.strip() for ln in f if ln.strip()]
        if names:
            return names
    print("⚠ class_names.txt missing or empty, using fallback list")
    return CLASS_NAMES_FALLBACK

def preprocess(frame: np.ndarray) -> np.ndarray:
    """Preprocess frame for ONNX model input - matches gesture model."""
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32) / 255.0
    if USE_IMAGENET_NORM:
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img = (img - mean) / std
    else:
        img = (img - 0.5) / 0.5
    img = np.transpose(img, (2, 0, 1)).astype(np.float32)
    img = np.expand_dims(img, axis=0)
    return img


class ONNXClassifier:
    """ONNX Runtime classifier."""
    
    def __init__(self, model_path: str, class_names: list[str]):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        self.class_names = class_names
        self.session = ort.InferenceSession(
            model_path,
            providers=["CPUExecutionProvider"],
        )
        
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self._debug_count = 0
        
        print(f"✓ Model loaded: {os.path.basename(model_path)}")
        print(f"✓ Input: {self.input_name}, shape: {self.session.get_inputs()[0].shape}")
        print(f"✓ Output: {self.output_name}, shape: {self.session.get_outputs()[0].shape}")
        print(f"✓ Normalization: {'ImageNet' if USE_IMAGENET_NORM else '(x/255.0 - 0.5) / 0.5'}")
        
        # Detect number of classes from output shape
        output_shape = self.session.get_outputs()[0].shape
        num_classes = output_shape[-1] if len(output_shape) > 1 else output_shape[0]
        print(f"✓ Detected {num_classes} output classes")
        
        if num_classes != len(self.class_names):
            print(f"⚠ WARNING: Model has {num_classes} classes but CLASS_NAMES has {len(self.class_names)}")
            print(f"  Please update CLASS_NAMES in the configuration!")

    def predict(self, frame: np.ndarray) -> Tuple[str, int, float, np.ndarray, float]:
        """Run inference and return prediction."""
        x = preprocess(frame)
        
        outputs = self.session.run([self.output_name], {self.input_name: x})
        logits = outputs[0][0]
        
        # Convert logits to probabilities if needed
        if np.all(logits >= 0) and np.all(logits <= 1) and 0.9 <= np.sum(logits) <= 1.1:
            probs = logits
        else:
            exp_logits = np.exp(logits - np.max(logits))
            probs = exp_logits / np.sum(exp_logits)
        
        idx = int(np.argmax(probs))
        
        # Handle case where model has more classes than CLASS_NAMES
        if idx < len(self.class_names):
            predicted_class = self.class_names[idx]
        else:
            predicted_class = f"class_{idx}"
        
        confidence = float(probs[idx])
        
        sorted_probs = np.sort(probs)[::-1]
        confidence_diff = sorted_probs[0] - sorted_probs[1] if len(sorted_probs) > 1 else confidence
        
        # Debug output
        self._debug_count += 1
        if self._debug_count % 30 == 0:
            print(f"\n[DEBUG] Frame {self._debug_count}")
            if len(probs) == len(self.class_names):
                print(f"  Probs: {dict(zip(self.class_names, probs))}")
            else:
                print(f"  Probs: {probs}")
            print(f"  Predicted: Gesture {predicted_class} ({confidence:.2%})")
            print(f"  Threshold: {DEFAULT_CONFIDENCE_THRESHOLD:.0%} | Diff: {confidence_diff:.3f}")
            will_move = confidence >= DEFAULT_CONFIDENCE_THRESHOLD and confidence_diff >= MIN_CONFIDENCE_DIFF
            print(f"  Will act: {'YES ✓' if will_move else 'NO ✗'}")
            print(f"  Input range: [{x.min():.3f}, {x.max():.3f}]")
            print(f"  Input shape: {x.shape}\n")
        
        return predicted_class, idx, confidence, probs, confidence_diff


class HazardRobotController:
    """Robot controller based on hazard detection."""
    
    def __init__(self, robot: Optional['AUPPBot']):
        self.robot = robot
        self.enabled = robot is not None
        self.lock = Lock()
        self.is_acting = False
        self.last_action_time = 0
        self.action_cooldown = 2.5  # Wait 2.5s after action completes

    def stop(self):
        """Stop all motors."""
        with self.lock:
            if self.enabled:
                try:
                    self.robot.stop_all()
                except Exception as e:
                    print(f"⚠ Error stopping: {e}")
            else:
                print("[SIM] STOP")

    def apply(self, detected_class: str):
        """Apply movement command based on detected class."""
        # Get action from ACTION_MAP
        action = ACTION_MAP.get(detected_class, "stop")
        
        if action == "stop":
            self.stop()
            return
        
        current_time = time.time()
        
        with self.lock:
            if self.is_acting:
                print(f"  ⏸ Skipping {action.upper()} - action already in progress")
                return
            time_since_last = current_time - self.last_action_time
            if time_since_last < self.action_cooldown:
                print(f"  ⏸ Skipping {action.upper()} - cooldown ({time_since_last:.2f}s)")
                return
        
        if not self.enabled:
            print(f"[SIM] {action.upper()} - would move for {MOVEMENT_TIME}s")
            return
        
        print(f"  🚀 Starting action thread for {action.upper()}")
        
        with self.lock:
            self.is_acting = True
        
        action_thread = Thread(target=self._execute_action, args=(action,), daemon=True)
        action_thread.start()
    
    def _execute_action(self, action: str):
        """Execute robot movement action."""
        try:
            print(f"🤖 Action: {action.upper()} - Moving for {MOVEMENT_TIME}s...")
            print(f"  Robot enabled: {self.enabled}")
            
            if not self.enabled:
                print(f"  ⚠ Robot not enabled - running in simulation mode")
                if action == "right_then_forward":
                    print(f"[SIM] Would TURN RIGHT then FORWARD")
                elif action == "left_then_forward":
                    print(f"[SIM] Would TURN LEFT then FORWARD")
                else:
                    print(f"[SIM] Would {action.upper()} for {MOVEMENT_TIME}s")
                time.sleep(MOVEMENT_TIME)
                return
            
            if self.robot is None:
                print(f"  ⚠ Robot object is None!")
                return
            
            # Execute the actual movement
            if action == "forward":
                print(f"  → Moving FORWARD at speed {DRIVE_SPEED}")
                try:
                    self.robot.motor1.forward(DRIVE_SPEED)
                    self.robot.motor2.forward(DRIVE_SPEED)
                    self.robot.motor3.forward(DRIVE_SPEED)
                    self.robot.motor4.forward(DRIVE_SPEED)
                    print(f"  ✓ Motors set to forward")
                    time.sleep(MOVEMENT_TIME)
                except Exception as e:
                    print(f"  ✗ Error setting motors forward: {e}")
                    raise
                    
            elif action == "backward":
                print(f"  → Moving BACKWARD at speed {DRIVE_SPEED}")
                try:
                    self.robot.motor1.backward(DRIVE_SPEED)
                    self.robot.motor2.backward(DRIVE_SPEED)
                    self.robot.motor3.backward(DRIVE_SPEED)
                    self.robot.motor4.backward(DRIVE_SPEED)
                    print(f"  ✓ Motors set to backward")
                    time.sleep(MOVEMENT_TIME)
                except Exception as e:
                    print(f"  ✗ Error setting motors backward: {e}")
                    raise
                    
            elif action == "right_then_forward":
                print(f"  → Turning RIGHT then FORWARD")
                try:
                    # Turn right
                    print(f"    Step 1: Turning RIGHT at speed {TURN_SPEED}")
                    self.robot.motor1.forward(TURN_SPEED)
                    self.robot.motor2.forward(TURN_SPEED)
                    self.robot.motor3.backward(TURN_SPEED)
                    self.robot.motor4.backward(TURN_SPEED)
                    print(f"    ✓ Turn motors set")
                    time.sleep(TURN_TIME_90)
                    # Stop before changing direction
                    self.robot.stop_all()
                    print(f"    ✓ Stopped after turn")
                    time.sleep(0.1)
                    # Move forward
                    print(f"    Step 2: Moving FORWARD at speed {DRIVE_SPEED}")
                    self.robot.motor1.forward(DRIVE_SPEED)
                    self.robot.motor2.forward(DRIVE_SPEED)
                    self.robot.motor3.forward(DRIVE_SPEED)
                    self.robot.motor4.forward(DRIVE_SPEED)
                    print(f"    ✓ Forward motors set")
                    time.sleep(MOVEMENT_TIME)
                except Exception as e:
                    print(f"  ✗ Error in right_then_forward: {e}")
                    raise
                    
            elif action == "left_then_forward":
                print(f"  → Turning LEFT then FORWARD")
                try:
                    # Turn left
                    print(f"    Step 1: Turning LEFT at speed {TURN_SPEED}")
                    self.robot.motor1.backward(TURN_SPEED)
                    self.robot.motor2.backward(TURN_SPEED)
                    self.robot.motor3.forward(TURN_SPEED)
                    self.robot.motor4.forward(TURN_SPEED)
                    print(f"    ✓ Turn motors set")
                    time.sleep(TURN_TIME_90)
                    # Stop before changing direction
                    self.robot.stop_all()
                    print(f"    ✓ Stopped after turn")
                    time.sleep(0.1)
                    # Move forward
                    print(f"    Step 2: Moving FORWARD at speed {DRIVE_SPEED}")
                    self.robot.motor1.forward(DRIVE_SPEED)
                    self.robot.motor2.forward(DRIVE_SPEED)
                    self.robot.motor3.forward(DRIVE_SPEED)
                    self.robot.motor4.forward(DRIVE_SPEED)
                    print(f"    ✓ Forward motors set")
                    time.sleep(MOVEMENT_TIME)
                except Exception as e:
                    print(f"  ✗ Error in left_then_forward: {e}")
                    raise
            
            # Stop all motors
            print(f"  → Stopping all motors")
            try:
                self.robot.stop_all()
                print(f"  ✓ All motors stopped")
            except Exception as e:
                print(f"  ✗ Error stopping motors: {e}")
                
        except Exception as e:
            print(f"⚠ Error performing action: {e}")
            import traceback
            traceback.print_exc()
            if self.enabled and self.robot:
                try:
                    print(f"  Attempting emergency stop...")
                    self.robot.stop_all()
                except Exception as stop_error:
                    print(f"  ✗ Emergency stop also failed: {stop_error}")
        finally:
            with self.lock:
                self.is_acting = False
                self.last_action_time = time.time()
                print(f"  ✓ Action completed, cooldown started")


class Camera:
    """Threaded camera."""
    
    def __init__(self, index: int = 0, width: int = 640, height: int = 480):
        self.cap = None
        try:
            if hasattr(cv2, 'CAP_V4L2'):
                self.cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
        except Exception:
            pass
        
        if self.cap is None or not self.cap.isOpened():
            self.cap = cv2.VideoCapture(index)
        
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open camera {index}")
        
        ret, _ = self.cap.read()
        if not ret:
            self.cap.release()
            raise RuntimeError(f"Camera {index} opened but can't read frames")
        
        if width:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        if height:
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        
        self.ok, self.frame = self.cap.read()
        self.lock = Lock()
        self.running = True
        
        self.thread = Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _loop(self):
        """Capture loop."""
        while self.running:
            ok, frame = self.cap.read()
            if ok:
                with self.lock:
                    self.ok = ok
                    self.frame = frame
            else:
                time.sleep(0.01)

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read latest frame."""
        with self.lock:
            return self.ok, None if self.frame is None else self.frame.copy()

    def release(self):
        """Release camera."""
        self.running = False
        time.sleep(0.05)
        self.cap.release()


# ========== INITIALIZATION ==========
print("\n" + "=" * 60)
print("🖐️ Gesture Control Robot (ONNX)")
print("=" * 60)

print("Connecting to robot...")
_robot = None
ROBOT_ENABLED = False
if AUPPBOT_AVAILABLE:
    # Check if port exists
    import os
    if not os.path.exists(ROBOT_PORT):
        print(f"⚠ Port {ROBOT_PORT} does not exist!")
        print(f"  Checking for available USB serial ports...")
        import glob
        usb_ports = glob.glob("/dev/ttyUSB*")
        if usb_ports:
            print(f"  Found USB ports: {usb_ports}")
            print(f"  Try updating ROBOT_PORT in the code if needed")
        else:
            print(f"  No USB serial ports found")
        print(f"  Running in simulation mode")
    else:
        try:
            print(f"  Attempting to connect to {ROBOT_PORT}...")
            _robot = AUPPBot(port=ROBOT_PORT, baud=115200, auto_safe=True)
            ROBOT_ENABLED = True
            print(f"✓ Robot connected on {ROBOT_PORT}")
            print(f"  Testing motor connection...")
            try:
                # Quick test - set motor to low speed and stop immediately
                _robot.motor1.forward(5)
                time.sleep(0.1)
                _robot.stop_all()
                print(f"  ✓ Motor test successful")
            except Exception as test_e:
                print(f"  ⚠ Motor test failed: {test_e}")
                print(f"  Continuing anyway...")
        except Exception as e:
            print(f"⚠ Robot connection failed: {e}")
            print(f"  Error type: {type(e).__name__}")
            import traceback
            traceback.print_exc()
            print(f"  Running in simulation mode")
            _robot = None
            ROBOT_ENABLED = False
else:
    print("⚠ auppbot module not available - simulation mode only")

print("\nLoading model...")
try:
    class_names = load_class_names()
    classifier = ONNXClassifier(MODEL_PATH, class_names)
except Exception as e:
    print(f"✗ Model failed: {e}")
    raise

controller = HazardRobotController(_robot)

print("\nInitializing camera...")
import warnings
warnings.filterwarnings('ignore')
try:
    cam = Camera(CAM_INDEX, width=CAMERA_WIDTH, height=CAMERA_HEIGHT)
    print(f"✓ Camera {CAM_INDEX} ready")
except Exception as e:
    warnings.resetwarnings()
    print(f"✗ Camera failed: {e}")
    raise
warnings.resetwarnings()

app = Flask(__name__)

state = {
    "last_class": None,
    "last_class_idx": None,
    "last_confidence": 0.0,
    "last_probs": None,
    "last_confidence_diff": 0.0,
    "inference_count": 0,
}

INDEX_HTML = """<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Gesture Control Robot</title>
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <style>
    * { margin: 0; padding: 0; box-sizing: border-box; }
    body { 
      display: grid; place-items: center; min-height: 100vh;
      background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
      color: #eaf0f6;
      font-family: 'Segoe UI', system-ui, sans-serif; padding: 20px;
    }
    .container {
      width: min(96vw, 1000px);
      background: rgba(17, 20, 23, 0.92);
      border-radius: 24px; padding: 28px;
      border: 1px solid rgba(255, 255, 255, 0.08);
      box-shadow: 0 25px 80px rgba(0, 0, 0, 0.6);
      backdrop-filter: blur(10px);
    }
    .header {
      display: flex; justify-content: space-between;
      align-items: center; margin-bottom: 24px;
    }
    h1 { 
      font-size: 1.6rem; font-weight: 700;
      background: linear-gradient(90deg, #ff6b6b, #feca57);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
    }
    .btn {
      border: 1px solid rgba(255, 107, 107, 0.3);
      background: rgba(255, 107, 107, 0.15);
      color: #ff6b6b; padding: 12px 24px;
      border-radius: 12px; cursor: pointer;
      font-weight: 600; transition: all 0.3s;
    }
    .btn:hover { 
      background: rgba(255, 107, 107, 0.3);
      transform: translateY(-2px);
    }
    .video-frame {
      width: 100%; aspect-ratio: 4/3;
      background: #0d1117; border-radius: 20px;
      overflow: hidden; border: 2px solid rgba(255, 107, 107, 0.2);
      box-shadow: 0 10px 40px rgba(0, 0, 0, 0.4);
      position: relative;
    }
    .video-frame img {
      width: 100%; height: 100%; object-fit: contain;
      display: block;
    }
    .video-frame .loading {
      position: absolute;
      top: 50%; left: 50%;
      transform: translate(-50%, -50%);
      color: #feca57;
      font-size: 1.2rem;
    }
    .stats-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
      gap: 14px; margin-top: 24px;
    }
    .stat-card {
      background: linear-gradient(145deg, rgba(40, 44, 52, 0.9), rgba(30, 34, 42, 0.9));
      padding: 18px; border-radius: 16px;
      text-align: center;
      border: 1px solid rgba(255, 255, 255, 0.06);
      transition: transform 0.3s;
    }
    .stat-card:hover { transform: translateY(-4px); }
    .stat-label {
      font-size: 0.7rem; opacity: 0.6;
      text-transform: uppercase; margin-bottom: 10px;
      letter-spacing: 1px;
    }
    .stat-value {
      font-size: 1.3rem; font-weight: 700;
      color: #feca57;
    }
    .stat-value.success { color: #26de81; }
    .stat-value.error { color: #ff6b6b; }
    .stat-value.warning { color: #feca57; }
  </style>
</head>
<body>
  <div class="container">
    <div class="header">
      <h1>🖐️ Gesture Control Robot</h1>
      <button class="btn" onclick="reloadStream()">🔄 Reload</button>
    </div>
    <div class="video-frame">
      <div class="loading" id="loading">Loading camera feed...</div>
      <img id="stream" src="/stream" alt="Live Feed" onload="hideLoading()" onerror="showError()">
    </div>
    <div class="stats-grid">
      <div class="stat-card">
        <div class="stat-label">Status</div>
        <div class="stat-value" id="health">...</div>
      </div>
      <div class="stat-card">
        <div class="stat-label">Robot</div>
        <div class="stat-value """ + ("success" if ROBOT_ENABLED else "error") + """">
          """ + ("✓ ACTIVE" if ROBOT_ENABLED else "✗ SIM") + """
        </div>
      </div>
      <div class="stat-card">
        <div class="stat-label">Detected Gesture</div>
        <div class="stat-value" id="detection">-</div>
      </div>
      <div class="stat-card">
        <div class="stat-label">Confidence</div>
        <div class="stat-value" id="confidence">0%</div>
      </div>
    </div>
  </div>
  <script>
    function hideLoading() {
      const loading = document.getElementById('loading');
      if (loading) loading.style.display = 'none';
    }
    function showError() {
      const loading = document.getElementById('loading');
      if (loading) {
        loading.textContent = 'Camera feed error - check connection';
        loading.style.display = 'block';
      }
    }
    async function updateStats() {
      try {
        const r = await fetch('/health', { cache: 'no-store' });
        const d = await r.json();
        document.getElementById('health').textContent = d.camera_ok ? '✓ OK' : '✗ ERROR';
        document.getElementById('health').className = 'stat-value ' + (d.camera_ok ? 'success' : 'error');
        if (d.detected_class) {
          // Show the detected number clearly
          const num = d.detected_class === 'BACKGROUND' ? 'BG' : d.detected_class;
          document.getElementById('detection').textContent = num;
          // Color coding based on number
          let cls = 'warning';
          if (d.detected_class === '1') cls = 'success';  // Green for forward
          else if (d.detected_class === '0' || d.detected_class === 'BACKGROUND') cls = 'error';  // Red for stop
          else if (d.detected_class === '2') cls = 'warning';  // Yellow for backward
          else if (d.detected_class === '3' || d.detected_class === '4') cls = 'warning';  // Yellow for turns
          document.getElementById('detection').className = 'stat-value ' + cls;
        } else {
          document.getElementById('detection').textContent = '-';
          document.getElementById('detection').className = 'stat-value';
        }
        const confPercent = (d.confidence * 100).toFixed(1);
        document.getElementById('confidence').textContent = confPercent + '%';
        // Color confidence based on threshold
        const confEl = document.getElementById('confidence');
        if (d.confidence >= 0.75) {
          confEl.className = 'stat-value success';
        } else if (d.confidence >= 0.5) {
          confEl.className = 'stat-value warning';
        } else {
          confEl.className = 'stat-value error';
        }
      } catch(e) {
        document.getElementById('health').textContent = '✗ OFFLINE';
        document.getElementById('health').className = 'stat-value error';
      }
    }
    function reloadStream() {
      const streamImg = document.getElementById('stream');
      const loading = document.getElementById('loading');
      if (loading) loading.style.display = 'block';
      streamImg.src = '/stream?t=' + Date.now();
    }
    // Auto-reload stream if it stops updating
    let lastUpdateTime = Date.now();
    const streamImg = document.getElementById('stream');
    streamImg.addEventListener('load', function() {
      lastUpdateTime = Date.now();
      hideLoading();
    });
    setInterval(function() {
      if (Date.now() - lastUpdateTime > 5000) {
        console.log('Stream appears stalled, reloading...');
        reloadStream();
      }
    }, 2000);
    updateStats();
    setInterval(updateStats, 500);
  </script>
</body>
</html>
"""


def generate_mjpeg_stream():
    """Generate MJPEG stream."""
    frame_count = 0
    
    while True:
        ok, frame = cam.read()
        if not ok or frame is None:
            time.sleep(0.02)
            continue
        
        if FLIP_CAMERA:
            # Horizontal flip (mirror) to match typical webcam preview
            frame = cv2.flip(frame, 1)
        
        annotated = frame.copy()
        frame_count += 1
        
        if frame_count % SKIP_FRAMES == 0:
            try:
                detected_class, class_idx, conf, probs, conf_diff = classifier.predict(frame)
                
                state["last_class"] = detected_class
                state["last_class_idx"] = class_idx
                state["last_confidence"] = conf
                state["last_probs"] = probs
                state["last_confidence_diff"] = conf_diff
                state["inference_count"] += 1
                
                is_confident = (conf >= DEFAULT_CONFIDENCE_THRESHOLD and 
                               conf_diff >= MIN_CONFIDENCE_DIFF)
                
                if is_confident:
                    print(f"✅ CONFIDENT: Gesture {detected_class} at {conf:.1%}")
                    print(f"  Robot enabled: {ROBOT_ENABLED}, Applying action...")
                    controller.apply(detected_class)
                    
                    action = ACTION_MAP.get(detected_class, "stop")
                    if action == "forward":
                        color = (0, 255, 0)
                    elif action == "backward":
                        color = (0, 165, 255)
                    elif action == "right_then_forward":
                        color = (255, 255, 0)
                    elif action == "left_then_forward":
                        color = (255, 0, 255)
                    else:
                        color = (0, 0, 255)
                    
                    label = f"Gesture: {detected_class} | Conf: {conf:.0%} | Action: {action.upper()}"
                else:
                    controller.stop()
                    color = (128, 128, 128)
                    label = f"Gesture: {detected_class} | Conf: {conf:.0%} (need {DEFAULT_CONFIDENCE_THRESHOLD:.0%})"
                
                cv2.putText(annotated, label, (15, 40),
                           cv2.FONT_HERSHEY_DUPLEX, 0.8, color, 2, cv2.LINE_AA)
                
                # Show probabilities for all classes
                class_list = getattr(classifier, "class_names", CLASS_NAMES_FALLBACK)
                if len(probs) == len(class_list):
                    prob_text = " ".join([f"{class_list[i]}:{probs[i]:.2f}" 
                                        for i in range(len(class_list))])
                else:
                    prob_text = " ".join([f"c{i}:{probs[i]:.2f}" 
                                        for i in range(len(probs))])
                cv2.putText(annotated, prob_text, (15, 80),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                
            except Exception as e:
                print(f"⚠ Inference error: {e}")
                import traceback
                traceback.print_exc()
                controller.stop()
        else:
            if state["last_class"]:
                conf = state["last_confidence"]
                conf_diff = state.get("last_confidence_diff", 0.0)
                detected_class = state["last_class"]
                is_confident = (conf >= DEFAULT_CONFIDENCE_THRESHOLD and 
                               conf_diff >= MIN_CONFIDENCE_DIFF)
                
                if is_confident:
                    action = ACTION_MAP.get(detected_class, "stop")
                    if action == "forward":
                        color = (0, 255, 0)
                    elif action == "backward":
                        color = (0, 165, 255)
                    elif action == "right_then_forward":
                        color = (255, 255, 0)
                    elif action == "left_then_forward":
                        color = (255, 0, 255)
                    else:
                        color = (0, 0, 255)
                    label = f"Gesture: {detected_class} | Conf: {conf:.0%} | Action: {action.upper()}"
                else:
                    controller.stop()
                    color = (128, 128, 128)
                    label = f"Gesture: {detected_class} | Conf: {conf:.0%} (need {DEFAULT_CONFIDENCE_THRESHOLD:.0%})"
                
                cv2.putText(annotated, label, (15, 40),
                           cv2.FONT_HERSHEY_DUPLEX, 0.8, color, 2, cv2.LINE_AA)
        
        status = "ROBOT: " + ("ACTIVE" if ROBOT_ENABLED else "SIM")
        cv2.putText(annotated, status, (15, annotated.shape[0] - 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                   (0, 255, 0) if ROBOT_ENABLED else (128, 128, 128), 2)
        
        # Encode frame as JPEG
        ok, jpeg = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        if not ok:
            continue
        
        jpeg_bytes = jpeg.tobytes()
        
        # MJPEG stream format
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n"
               b"Content-Length: " + str(len(jpeg_bytes)).encode() + b"\r\n"
               b"\r\n" + jpeg_bytes + b"\r\n")


@app.route("/")
def index():
    return make_response(INDEX_HTML, 200)


@app.route("/health")
def health():
    ok, _ = cam.read()
    return jsonify({
        "camera_ok": bool(ok),
        "robot_enabled": ROBOT_ENABLED,
        "detected_class": state["last_class"],
        "detected_class_idx": state.get("last_class_idx"),
        "confidence": state["last_confidence"],
        "inference_count": state["inference_count"],
    })


@app.route("/stream")
def stream():
    """MJPEG video stream endpoint."""
    return Response(
        generate_mjpeg_stream(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0"
        }
    )


@app.route("/api/detection")
def api_detection():
    return jsonify({
        "detected_class": state["last_class"],
        "confidence": state["last_confidence"],
        "probabilities": state["last_probs"].tolist() if state["last_probs"] is not None else None,
        "robot_enabled": ROBOT_ENABLED,
        "class_names": getattr(classifier, "class_names", CLASS_NAMES_FALLBACK),
    })


def main():
    import socket
    def get_local_ip():
        """Get the local IP address."""
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except:
            return "localhost"
    
    local_ip = get_local_ip()
    
    print("\n" + "=" * 60)
    print("Configuration:")
    print(f"  Model: {os.path.basename(MODEL_PATH)}")
    print(f"  Classes: {getattr(classifier, 'class_names', CLASS_NAMES_FALLBACK)}")
    print(f"  Action Map: {ACTION_MAP}")
    print(f"  Camera: {CAM_INDEX}")
    print(f"  Robot: {'ENABLED' if ROBOT_ENABLED else 'DISABLED (simulation)'}")
    print(f"  Confidence threshold: {DEFAULT_CONFIDENCE_THRESHOLD:.0%}")
    print(f"  Movement time: {MOVEMENT_TIME}s (forward/back), {TURN_TIME_90}s (turns)")
    print(f"\n🌐 Web UI (Local): http://localhost:5000")
    print(f"🌐 Web UI (Network): http://{local_ip}:5000")
    print("=" * 60 + "\n")
    
    try:
        app.run(host="0.0.0.0", port=5000, threaded=True, debug=False)
    except KeyboardInterrupt:
        print("\n\nShutting down...")
    finally:
        print("Cleaning up...")
        controller.stop()
        cam.release()
        if ROBOT_ENABLED and _robot:
            _robot.close()
        print("✓ Done")


if __name__ == "__main__":
    main()

