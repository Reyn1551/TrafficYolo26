"""
Sistem Deteksi Kendaraan Real-time Khusus ATCS (Optimized)
=======================================================================================
Mode: Direct Link (Tanpa Menu)
Sumber: ATCS CCTV Stream
Optimasi: Kalman Filter, Multi-threading, GPU Support, Perspective Transform Speed
"""

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from collections import defaultdict, deque
import time
from datetime import datetime
import math
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
import threading
import queue
import json
import os
from scipy.optimize import linear_sum_assignment
from filterpy.kalman import KalmanFilter
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Disable specific YOLO logging
import logging
logging.getLogger("ultralytics").setLevel(logging.ERROR)


# ================= KONFIGURASI ATCS (UBAH DISINI) =================

# Link Stream ATCS 
ATCS_URL = "http://cctvjss.jogjakota.go.id/atcs/ATCS_Lampu_Merah_SugengJeroni2.stream/playlist.m3u8"

# Nama file model
MODEL_FILE = 'yolov26n.pt' 

# Target FPS untuk sistem
TARGET_FPS = 30.0  # Set reasonable target for CPU/Stream limit

# --- KALIBRASI PERSPEKTIF (PENTING UNTUK SPEED) ---
# 4 Titik di layar (Source) yang membentuk persegi panjang di dunia nyata
# Urutan: Top-Left, Top-Right, Bottom-Right, Bottom-Left
# Contoh default ini harus disesuaikan dengan tampilan CCTV spesifik!
SOURCE_POINTS = np.float32([
    [450, 250],  # Top-Left (x, y)
    [850, 250],  # Top-Right
    [1000, 600], # Bottom-Right
    [200, 600]   # Bottom-Left
])

# Ukuran persegi panjang di dunia nyata (meter)
# Misal kita ambil area jalan lebar 7 meter (2 lajur) dan panjang ke belakang 20 meter
REAL_WORLD_WIDTH = 7.0   # Meter
REAL_WORLD_LENGTH = 20.0 # Meter

# --- OPTIMASI PERFORM ---
ENABLE_MOTION_DETECTION = False    # Disabled for debugging low-light
RESIZE_WIDTH = 1280            # Set None jika ingin resolusi asli (Not Recommended for 4K)
RESIZE_HEIGHT = 720
DEBUG_MODE = True              # Enable detailed performance logging
SKIP_INFERENCE_FRAMES = 1      # Run EVERY frame for debugging
USE_HALF_PRECISION = True      # Use FP16 if GPU supported

def log_debug(msg):
    if DEBUG_MODE:
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        print(f"[{timestamp}] [DEBUG] {msg}")



# ==================================================================

# ==================== LAYER 1: PERSPECTIVE TRANSFORMER (NEW) ====================
class PerspectiveTransformer:
    def __init__(self, src_points, real_width, real_length):
        self.src_points = src_points
        self.dst_points = np.float32([
            [0, 0],
            [real_width, 0],
            [real_width, real_length],
            [0, real_length]
        ])
        self.M = cv2.getPerspectiveTransform(self.src_points, self.dst_points)
        print("Perspective Matrix Calculated")

    def transform_point(self, point):
        # Point is (x, y)
        p = np.array([[[point[0], point[1]]]], dtype=np.float32)
        dst = cv2.perspectiveTransform(p, self.M)
        return dst[0][0] # Returns (x, y) in meters

    def calculate_speed(self, p1, p2, time_elapsed):
        # Euclidean distance in meters
        dist = math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
        if time_elapsed <= 0: return 0.0
        speed_ms = dist / time_elapsed
        return speed_ms * 3.6 # km/h

# ==================== LAYER 2: KALMAN FILTER TRACKER (REFACTORED) ====================

class KalmanBoxTracker:
    count = 0
    def __init__(self, bbox: Tuple[int, int, int, int], class_name: str, confidence: float):
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        # State: [u, v, s, r, u_dot, v_dot, s_dot]
        # u, v: center x, y
        # s: area scale
        # r: aspect ratio
        self.kf.F = np.array([
            [1,0,0,0,1,0,0],
            [0,1,0,0,0,1,0],
            [0,0,1,0,0,0,1],
            [0,0,0,1,0,0,0],
            [0,0,0,0,1,0,0],
            [0,0,0,0,0,1,0],
            [0,0,0,0,0,0,1]
        ])
        self.kf.H = np.array([
            [1,0,0,0,0,0,0],
            [0,1,0,0,0,0,0],
            [0,0,1,0,0,0,0],
            [0,0,0,1,0,0,0]
        ])
        self.kf.R[2:,2:] *= 10.
        self.kf.P[4:,4:] *= 1000.
        self.kf.P *= 10.
        self.kf.Q[-1,-1] *= 0.01
        self.kf.Q[4:,4:] *= 0.01

        self.kf.x[:4] = self.convert_bbox_to_z(bbox)
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        
        self.class_name = class_name
        self.confidence = confidence
        self.time_since_update = 0
        self.history = deque(maxlen=20)
        self.last_position_meters = None
        self.current_speed = 0.0
        self.hits = 0
        self.hit_streak = 0
        self.age = 0

    def convert_bbox_to_z(self, bbox):
        x1, y1, x2, y2 = bbox
        w = x2 - x1
        h = y2 - y1
        x = x1 + w/2.
        y = y1 + h/2.
        s = w * h
        r = w / float(h)
        return np.array([x, y, s, r]).reshape((4, 1))

    def convert_x_to_bbox(self, x, score=None):
        w = np.sqrt(x[2] * x[3])
        h = x[2] / w
        return np.array([x[0] - w/2., x[1] - h/2., x[0] + w/2., x[1] + h/2.]).reshape((1,4))

    def update(self, bbox, confidence=None):
        self.time_since_update = 0
        self.history.append(self.convert_x_to_bbox(self.kf.x)[0])
        self.hits += 1
        self.hit_streak += 1
        if confidence: self.confidence = confidence
        self.kf.update(self.convert_bbox_to_z(bbox))

    def predict(self):
        if((self.kf.x[6]+self.kf.x[2])<=0):
            self.kf.x[6] *= 0.0
        self.kf.predict()
        self.age += 1
        if(self.time_since_update>0):
            self.hit_streak = 0
        self.time_since_update += 1
        return self.convert_x_to_bbox(self.kf.x)[0]

    def get_state(self):
        return self.convert_x_to_bbox(self.kf.x)[0]

class KalmanTrackerManager:
    def __init__(self, max_age=30, min_hits=3, iou_threshold=0.3):
        self.trackers = []
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold

    def update(self, detections):
        # Detections: list of {'bbox': [x1,y1,x2,y2], 'score': float, 'class': str}
        trks = np.zeros((len(self.trackers), 5))
        to_del = []
        ret = []
        
        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict()
            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
            if np.any(np.isnan(pos)):
                to_del.append(t)
                
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks)) 
        for t in reversed(to_del):
            self.trackers.pop(t)
            
        # If no detections (e.g. skipped frame), just return predicted tracks
        if len(detections) == 0:
            # Only return valid active tracks
            i = len(self.trackers)
            for trk in reversed(self.trackers):
                 if (trk.time_since_update < 5) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                     ret.append(trk)
                 i -= 1
                 if(trk.time_since_update > self.max_age):
                    self.trackers.pop(i)
            self.frame_count += 1
            return ret

        matched, unmatched_dets, unmatched_trks = self.associate_detections_to_trackers(detections, trks)

        # Update matched trackers
        for d, t in matched:
            self.trackers[t].update(detections[d]['bbox'], detections[d]['score'])

        # Create new trackers
        for i in unmatched_dets:
            trk = KalmanBoxTracker(detections[i]['bbox'], detections[i]['class'], detections[i]['score'])
            self.trackers.append(trk)

        i = len(self.trackers)
        for trk in reversed(self.trackers):
            d = trk.get_state()
            if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                ret.append(trk) 
            i -= 1
            if(trk.time_since_update > self.max_age):
                self.trackers.pop(i)

        self.frame_count += 1
        return ret

    def associate_detections_to_trackers(self, detections, trackers):
        if(len(trackers)==0):
            return np.empty((0,2),dtype=int), np.arange(len(detections)), np.empty((0,5),dtype=int)
        
        iou_matrix = np.zeros((len(detections), len(trackers)), dtype=np.float32)
        for d, det in enumerate(detections):
            for t, trk in enumerate(trackers):
                iou_matrix[d, t] = self.iou(det['bbox'], trk) # Simple IOU

        if min(iou_matrix.shape) > 0:
            a = (iou_matrix > self.iou_threshold).astype(np.int32)
            if a.sum(1).max() == 1 and a.sum(0).max() == 1:
                matched_indices = np.stack(np.where(a), axis=1)
            else:
                matched_indices = linear_sum_assignment(-iou_matrix)
                matched_indices = np.array(list(zip(*matched_indices)))
        else:
            matched_indices = np.empty((0,2))

        unmatched_detections = []
        for d, det in enumerate(detections):
            if(d not in matched_indices[:,0]):
                unmatched_detections.append(d)
                
        unmatched_trackers = []
        for t, trk in enumerate(trackers):
            if(t not in matched_indices[:,1]):
                unmatched_trackers.append(t)

        matches = []
        for m in matched_indices:
            if(iou_matrix[m[0], m[1]] < self.iou_threshold):
                unmatched_detections.append(m[0])
                unmatched_trackers.append(m[1])
            else:
                matches.append(m.reshape(1,2))
                
        if(len(matches)==0):
            matches = np.empty((0,2),dtype=int)
        else:
            matches = np.concatenate(matches, axis=0)

        return matches, unmatched_detections, unmatched_trackers

    def iou(self, bb_test, bb_gt):
        xx1 = np.maximum(bb_test[0], bb_gt[0])
        yy1 = np.maximum(bb_test[1], bb_gt[1])
        xx2 = np.minimum(bb_test[2], bb_gt[2])
        yy2 = np.minimum(bb_test[3], bb_gt[3])
        w = np.maximum(0., xx2 - xx1)
        h = np.maximum(0., yy2 - yy1)
        wh = w * h
        o = wh / ((bb_test[2]-bb_test[0])*(bb_test[3]-bb_test[1]) + 
            (bb_gt[2]-bb_gt[0])*(bb_gt[3]-bb_gt[1]) - wh)
        return(o)
    
    frame_count = 0

# ==================== LAYER 3: ROBUST STREAM LOADER ====================

class StreamLoader:
    def __init__(self, src, queue_size=128):
        self.src = src
        # Ganti capture options untuk ffmpeg
        # Ganti capture options untuk ffmpeg
        if os.name == 'posix':
            # Optimize for HLS/HTTP stream
            # buffer_size increased to 20MB (20 * 1024 * 1024)
            # timeout set to 30 seconds (30000000 us)
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "buffer_size;20971520|timeout;30000000"
        
        self.stream = cv2.VideoCapture(self.src)
        self.stopped = False
        self.queue_size = queue_size
        self.Q = deque(maxlen=queue_size)
        self.lock = threading.Lock()
        self.reconnecting = False
        
        if not self.stream.isOpened():
            print(f"[ERROR] Failed to open stream: {src}")

    def start(self):
        t = threading.Thread(target=self.update, args=())
        t.daemon = True
        t.start()
        return self
    
    def update(self):
        fails = 0
        while not self.stopped:
            if self.reconnecting:
                time.sleep(1)
                continue

            if not self.stream.isOpened():
                self._reconnect()
                continue
                
            grabbed, frame = self.stream.read()
            if not grabbed:
                fails += 1
                if fails > 5: # Retry faster (was 10)
                    print(f"[WARNING] Stream unstable (Empty Frames). Reconnecting...")
                    self._reconnect()
                    fails = 0
                else:
                    time.sleep(0.05) # Wait a bit longer between retries
                continue
            
            fails = 0
            with self.lock:
                self.Q.append(frame)
            
            # Maintain queue size manually if buffer fills up fast
            while len(self.Q) >= self.queue_size and not self.stopped:
                 with self.lock:
                    self.Q.popleft() # Drop oldest frame to keep realtime
            
    def _reconnect(self):
        self.reconnecting = True
        if self.stream.isOpened():
            self.stream.release()
        
        print("[INFO] Attempting to reconnect...")
        time.sleep(2) # Wait before reconnect
        self.stream = cv2.VideoCapture(self.src)
        
        if self.stream.isOpened():
            print("[INFO] Stream reconnected success!")
        else:
            print("[ERROR] Reconnection failed, retrying...")
        self.reconnecting = False
            
    def read(self):
        with self.lock:
            if len(self.Q) > 0:
                return True, self.Q.popleft()
            else:
                return False, None
    
    def running(self):
        return not self.stopped
        
    def stop(self):
        self.stopped = True
        if self.stream.isOpened():
            self.stream.release()

# ==================== LAYER 4: MOTION DETECTOR (OPTIMIZATION) ====================
class MotionDetector:
    def __init__(self, history=500, var_threshold=16, min_area=500):
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=history, varThreshold=var_threshold, detectShadows=False)
        self.min_area = min_area
        
    def has_motion(self, frame):
        # Resize for faster processing
        small_frame = cv2.resize(frame, (640, 360))
        fg_mask = self.bg_subtractor.apply(small_frame)
        _, thresh = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            if cv2.contourArea(contour) > self.min_area:
                return True
        return False

# ==================== LAYER 5: IMAGE ENHANCEMENT (Night Mode) ====================
class ImageEnhancer:
    def __init__(self, low_light_threshold=50, gamma=1.5):
        self.mean_brightness = 0
        self.low_light_threshold = low_light_threshold
        self.gamma = gamma
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        self.is_night_mode = False
        self.gamma_table = self._build_gamma_table(gamma)

    def _build_gamma_table(self, gamma):
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype("uint8")
        return table

    def check_brightness(self, frame):
        # Convert to HSV to get brightness (V channel)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        self.mean_brightness = hsv[..., 2].mean()
        # if DEBUG_MODE: log_debug(f"Brightness: {self.mean_brightness:.2f}")
        
        # Hysteresis for stability
        prev_mode = self.is_night_mode
        if self.is_night_mode:
            if self.mean_brightness > self.low_light_threshold + 10:
                self.is_night_mode = False
        else:
            if self.mean_brightness < self.low_light_threshold:
                self.is_night_mode = True
        
        if prev_mode != self.is_night_mode and DEBUG_MODE:
            log_debug(f"Night Mode Toggled: {self.is_night_mode} (Brightness: {self.mean_brightness:.2f})")
                
        return self.is_night_mode

    def enhance(self, frame):
        if not self.is_night_mode:
            return frame

        # Apply Gamma Correction
        enhanced = cv2.LUT(frame, self.gamma_table)
        
        # Apply CLAHE to L channel of LAB
        lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = self.clahe.apply(l)
        enhanced = cv2.merge((l, a, b))
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        
        return enhanced

# ==================== MAIN SYSTEM ====================

class TrafficMonitorSystem:
    def __init__(self):
        print("Initializing Traffic Monitor System v2.0...")
        
        # 1. Hardware Check
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Device: {self.device}")
        
        # 2. Model Init
        try:
            self.model = YOLO(MODEL_FILE)
            self.model.to(self.device)
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Error loading model: {e}")
            exit(1)
            
        # 3. Tracker & Utils
        self.tracker_manager = KalmanTrackerManager(max_age=25, min_hits=3, iou_threshold=0.3)
        self.perspective_tx = PerspectiveTransformer(SOURCE_POINTS, REAL_WORLD_WIDTH, REAL_WORLD_LENGTH)
        self.stream_loader = None
        self.motion_detector = MotionDetector()
        self.enhancer = ImageEnhancer(low_light_threshold=60, gamma=1.5) # Initialize Enhancer
        
        # Stats
        self.vehicle_counts = defaultdict(int)
        self.id_history = set()
        self.prev_frame_count = 0 
        self.processing_fps = 0
        
    def run(self):
        self.stream_loader = StreamLoader(ATCS_URL).start()
        time.sleep(2.0) # Warmup
        
        prev_time = time.time()
        fps_monitor = 0
        
        while self.stream_loader.running():
            # --- PERFORMANCE PROFILING ---
            t_start = time.time()
            
            ret, frame = self.stream_loader.read()
            if not ret:
                time.sleep(0.005)
                continue
            
            t_read = time.time()
            if DEBUG_MODE and self.stream_loader.Q:
                log_debug(f"Queue Size: {len(self.stream_loader.Q)}")

            # --- RESIZE OPTIMIZATION ---
            if RESIZE_WIDTH is not None:
                frame = cv2.resize(frame, (RESIZE_WIDTH, RESIZE_HEIGHT))
            
            t_resize = time.time()

            # --- LOW LIGHT ENHANCEMENT ---
            is_night = self.enhancer.check_brightness(frame)
            if is_night:
                frame_viz = frame.copy() # Keep original for visualization if needed, but we usually want to see enhanced
                frame = self.enhancer.enhance(frame)
                if DEBUG_MODE: cv2.putText(frame, "NIGHT MODE", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            
            t_enhance = time.time()
            
            curr_time = time.time()
            dt = curr_time - prev_time
            prev_time = curr_time
            fps_monitor = 0.9 * fps_monitor + 0.1 * (1/dt if dt > 0 else 0)
            
            # --- INFERENCE CONTROL ---
            # Hybrid Tracking: Run YOLO every N frames, use Tracker Prediction for others
            do_inference = (self.prev_frame_count % SKIP_INFERENCE_FRAMES == 0)
            
            # Motion check override: If no motion, force skip inference to save power/heat
            if ENABLE_MOTION_DETECTION and not self.motion_detector.has_motion(frame):
                do_inference = False
                if DEBUG_MODE and do_inference: log_debug("Motion: NONE - Override Skip")
            
            t_motion = time.time()

            detections = []
            
            if do_inference:
                if DEBUG_MODE: log_debug(f"Starting Inference (Frame {self.prev_frame_count})...")
                results = self.model.predict(frame, conf=0.4, verbose=False, iou=0.4, half=USE_HALF_PRECISION)
                t_infer = time.time()
                
                for result in results:
                    for box in result.boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = float(box.conf[0].cpu())
                        cls = int(box.cls[0].cpu())
                        if cls in [2, 3, 5, 7]:
                            class_name = self.model.names[cls]
                            detections.append({
                                'bbox': [x1, y1, x2, y2],
                                'score': conf,
                                'class': class_name
                            })
                
                if DEBUG_MODE and len(detections) > 0:
                     log_debug(f"YOLO Detected: {len(detections)} objects")
            else:
                t_infer = time.time()
                # If skipping inference, detections list remains empty
                # The tracker manager handles empty detections by just predicting
                if DEBUG_MODE: log_debug(f"Skipping Inference (Track Only)...")

            
            # --- TRACKING ---
            # If detections is empty (skipped frame), tracker updates based on its internal prediction
            active_tracks = self.tracker_manager.update(detections)
            self.prev_frame_count += 1

            t_track = time.time()
            
            if DEBUG_MODE:
                log_debug(f"Timing: Read={(t_read-t_start)*1000:.1f}ms | Resize={(t_resize-t_read)*1000:.1f}ms | Enhance={(t_enhance-t_resize)*1000:.1f}ms | Motion={(t_motion-t_enhance)*1000:.1f}ms | Infer={(t_infer-t_motion)*1000:.1f}ms | Track={(t_track-t_infer)*1000:.1f}ms")
                log_debug(f"Detections: {len(detections)} | Tracks: {len(active_tracks)}")

            
            # --- VISUALIZATION & SPEED ---
            cv2.polylines(frame, [np.int32(SOURCE_POINTS)], True, (0, 255, 255), 2)
            
            for trk in active_tracks:
                bbox = trk.get_state()
                x1, y1, x2, y2 = map(int, bbox)
                
                # Bottom Center Point for homography
                cx = int((x1 + x2) / 2)
                cy = int(y2)
                
                # Project to Real World
                real_pos = self.perspective_tx.transform_point((cx, cy))
                
                # Update Speed
                if trk.last_position_meters is not None:
                    # Smoothing speed
                    inst_speed = self.perspective_tx.calculate_speed(trk.last_position_meters, real_pos, dt)
                    # Filter crazy speeds (noise)
                    if inst_speed < 150: # Cap at 150 km/h sanity check
                        trk.current_speed = 0.8 * trk.current_speed + 0.2 * inst_speed
                
                trk.last_position_meters = real_pos
                
                # Draw
                color = (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{trk.class_name} {int(trk.current_speed)} km/h", 
                           (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                # Counting logic (crossing line)
                # Example: Center of image Y
                mid_line = frame.shape[0] // 2
                if (cy > mid_line - 10) and (cy < mid_line + 10):
                    if trk.id not in self.id_history:
                        self.vehicle_counts[trk.class_name] += 1
                        self.id_history.add(trk.id)

            # --- HUD ---
            cv2.putText(frame, f"FPS: {int(fps_monitor)}", (20, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            y_off = 80
            for k, v in self.vehicle_counts.items():
                cv2.putText(frame, f"{k}: {v}", (20, y_off), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                y_off += 30

            cv2.imshow("ATCS Traffic Monitor", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        self.stream_loader.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    app = TrafficMonitorSystem()
    app.run()