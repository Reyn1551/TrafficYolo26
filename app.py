"""
Sistem Deteksi Kendaraan Real-time Khusus ATCS
=======================================================================================
Mode: Direct Link (Tanpa Menu)
Sumber: ATCS CCTV Stream
Optimasi: Kalman Filter, Multi-threading, GPU Support

Cara Pakai:
1. Pastikan file model (yolov8n.pt atau yolov26n.pt) ada di folder yang sama.
2. Jalankan script. Program akan langsung membuka stream.
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
warnings.filterwarnings('ignore')

# ================= KONFIGURASI ATCS (UBAH DISINI) =================

# Link Stream ATCS (Ganti link ini sesuai kebutuhan CCTV spesifik)
# Contoh: Simpang Wirosaban / Sugeng Jeroni
ATCS_URL = "http://cctvjss.jogjakota.go.id/atcs/ATCS_Simpang_Wirosaban_Barat.stream/playlist.m3u8"

# Nama file model (Pastikan file .pt ada di folder yang sama)
# Jika kamu punya model custom bernama yolov26n.pt, ubah string di bawah ini.
MODEL_FILE = 'yolov8n.pt' 

# Target FPS untuk sistem
TARGET_FPS = 60.0

# ==================================================================


# ==================== LAYER 1: KALMAN FILTER TRACKER ====================

class KalmanBoxTracker:
    count = 0
    
    def __init__(self, bbox: Tuple[int, int, int, int], class_name: str, confidence: float):
        self.kf = KalmanFilter(dim_x=8, dim_z=4)
        self.kf.F = np.array([
            [1, 0, 0, 0, 1, 0, 0, 0],
            [0, 1, 0, 0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0, 0, 1, 0],
            [0, 0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 1]
        ])
        self.kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0]
        ])
        self.kf.R[2:, 2:] *= 10.0
        self.kf.R *= 1.0
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01
        self.kf.Q *= 0.01
        self.kf.P[4:, 4:] *= 1000.0
        self.kf.P *= 10.0
        
        x1, y1, x2, y2 = bbox
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        w, h = x2 - x1, y2 - y1
        self.kf.x[:4] = np.array([cx, cy, w, h]).reshape(4, 1)
        
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.class_name = class_name
        self.confidence = confidence
        self.age = 0
        self.hits = 1
        self.hit_streak = 1
        self.time_since_update = 0
        self.history = deque(maxlen=50)
        self.speed_history = deque(maxlen=30)
        self.prediction_history = deque(maxlen=10)
        self.innovation = 0.0
        
    def predict(self) -> np.ndarray:
        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        cx, cy, w, h = self.kf.x[:4].flatten()
        x1, y1 = cx - w/2, cy - h/2
        x2, y2 = cx + w/2, cy + h/2
        self.prediction_history.append((int(cx), int(cy)))
        return np.array([x1, y1, x2, y2])
    
    def update(self, bbox: Tuple[int, int, int, int], confidence: float = None):
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1
        if confidence is not None:
            self.confidence = confidence
        x1, y1, x2, y2 = bbox
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        w, h = x2 - x1, y2 - y1
        measurement = np.array([cx, cy, w, h]).reshape(4, 1)
        predicted_measurement = self.kf.H @ self.kf.x
        self.innovation = np.linalg.norm(measurement - predicted_measurement)
        self.kf.update(measurement)
        self.history.append((int(cx), int(cy)))
    
    def get_state(self) -> Tuple[int, int, int, int]:
        cx, cy, w, h = self.kf.x[:4].flatten()
        x1, y1 = cx - w/2, cy - h/2
        x2, y2 = cx + w/2, cy + h/2
        return (int(x1), int(y1), int(x2), int(y2))
    
    def get_velocity(self) -> Tuple[float, float]:
        vx, vy = self.kf.x[4, 0], self.kf.x[5, 0]
        return float(vx), float(vy)
    
    def get_center(self) -> Tuple[int, int]:
        cx, cy = int(self.kf.x[0, 0]), int(self.kf.x[1, 0])
        return (cx, cy)


class KalmanTrackerManager:
    def __init__(self, max_age: int = 30, min_hits: int = 3, 
                 iou_threshold: float = 0.3, distance_threshold: float = 100):
        self.trackers: List[KalmanBoxTracker] = []
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.distance_threshold = distance_threshold
        self.frame_count = 0
        
    def iou_batch(self, bb_test: np.ndarray, bb_gt: np.ndarray) -> np.ndarray:
        bb_test = np.array(bb_test)
        bb_gt = np.array(bb_gt)
        if len(bb_test) == 0 or len(bb_gt) == 0:
            return np.zeros((len(bb_test), len(bb_gt)))
        xx1 = np.maximum(bb_test[:, 0].reshape(-1, 1), bb_gt[:, 0].reshape(1, -1))
        yy1 = np.maximum(bb_test[:, 1].reshape(-1, 1), bb_gt[:, 1].reshape(1, -1))
        xx2 = np.minimum(bb_test[:, 2].reshape(-1, 1), bb_gt[:, 2].reshape(1, -1))
        yy2 = np.minimum(bb_test[:, 3].reshape(-1, 1), bb_gt[:, 3].reshape(1, -1))
        w = np.maximum(0., xx2 - xx1)
        h = np.maximum(0., yy2 - yy1)
        intersection = w * h
        area_test = (bb_test[:, 2] - bb_test[:, 0]) * (bb_test[:, 3] - bb_test[:, 1])
        area_gt = (bb_gt[:, 2] - bb_gt[:, 0]) * (bb_gt[:, 3] - bb_gt[:, 1])
        union = area_test.reshape(-1, 1) + area_gt.reshape(1, -1) - intersection
        return intersection / (union + 1e-6)
    
    def distance_batch(self, centers_test: List[Tuple], centers_gt: List[Tuple]) -> np.ndarray:
        n, m = len(centers_test), len(centers_gt)
        if n == 0 or m == 0:
            return np.zeros((n, m))
        dist_matrix = np.zeros((n, m))
        for i, ct in enumerate(centers_test):
            for j, cg in enumerate(centers_gt):
                dist_matrix[i, j] = math.sqrt((ct[0] - cg[0])**2 + (ct[1] - cg[1])**2)
        return dist_matrix
    
    def associate_detections_to_trackers(self, detections: List[dict]) -> Tuple[List, List, List]:
        if len(self.trackers) == 0:
            return [], list(range(len(detections))), []
        predicted_boxes = np.array([t.predict() for t in self.trackers])
        predicted_centers = [t.get_center() for t in self.trackers]
        det_boxes = np.array([d['bbox'] for d in detections]) if detections else np.array([])
        det_centers = [d['center'] for d in detections]
        iou_matrix = self.iou_batch(det_boxes, predicted_boxes) if len(det_boxes) > 0 else np.zeros((0, len(self.trackers)))
        dist_matrix = self.distance_batch(det_centers, predicted_centers)
        max_dist = max(self.distance_threshold, dist_matrix.max() + 1) if dist_matrix.size > 0 else self.distance_threshold
        cost_matrix = (1 - iou_matrix) + (dist_matrix / max_dist) * 0.5
        matched_indices = linear_sum_assignment(cost_matrix)
        matches = []
        unmatched_detections = list(range(len(detections)))
        unmatched_trackers = list(range(len(self.trackers)))
        for d_idx, t_idx in zip(matched_indices[0], matched_indices[1]):
            if iou_matrix.size > 0 and d_idx < iou_matrix.shape[0] and t_idx < iou_matrix.shape[1]:
                if iou_matrix[d_idx, t_idx] >= self.iou_threshold or dist_matrix[d_idx, t_idx] <= self.distance_threshold:
                    matches.append((d_idx, t_idx))
                    if d_idx in unmatched_detections:
                        unmatched_detections.remove(d_idx)
                    if t_idx in unmatched_trackers:
                        unmatched_trackers.remove(t_idx)
        return matches, unmatched_detections, unmatched_trackers
    
    def update(self, detections: List[dict]) -> List[dict]:
        self.frame_count += 1
        matches, unmatched_dets, unmatched_trks = self.associate_detections_to_trackers(detections)
        for d_idx, t_idx in matches:
            det = detections[d_idx]
            self.trackers[t_idx].update(det['bbox'], det['confidence'])
        for d_idx in unmatched_dets:
            det = detections[d_idx]
            new_tracker = KalmanBoxTracker(det['bbox'], det['class'], det['confidence'])
            self.trackers.append(new_tracker)
        trackers_to_remove = []
        for t_idx, trk in enumerate(self.trackers):
            if trk.time_since_update > self.max_age:
                trackers_to_remove.append(t_idx)
        for t_idx in reversed(trackers_to_remove):
            self.trackers.pop(t_idx)
        results = []
        for trk in self.trackers:
            if trk.time_since_update == 0 and trk.hit_streak >= self.min_hits:
                bbox = trk.get_state()
                center = trk.get_center()
                vx, vy = trk.get_velocity()
                results.append({
                    'track_id': trk.id,
                    'class': trk.class_name,
                    'confidence': trk.confidence,
                    'bbox': bbox,
                    'center': center,
                    'velocity': (vx, vy),
                    'age': trk.age,
                    'hits': trk.hits,
                    'history': list(trk.history),
                    'innovation': trk.innovation
                })
        return results


# ==================== LAYER 2: MOTION DETECTION ====================

class MotionDetector:
    def __init__(self, history: int = 500, var_threshold: float = 16, 
                 detect_shadows: bool = False, min_area: int = 500):
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=history,
            varThreshold=var_threshold,
            detectShadows=detect_shadows
        )
        self.min_area = min_area
        self.motion_mask = None
        self.roi_regions = []
        
    def detect_motion(self, frame: np.ndarray) -> Tuple[np.ndarray, List[Tuple]]:
        fg_mask = self.bg_subtractor.apply(frame)
        _, thresh = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        motion_regions = []
        for contour in contours:
            if cv2.contourArea(contour) >= self.min_area:
                x, y, w, h = cv2.boundingRect(contour)
                x = max(0, x - 20)
                y = max(0, y - 20)
                w = min(frame.shape[1] - x, w + 40)
                h = min(frame.shape[0] - y, h + 40)
                motion_regions.append((x, y, x + w, y + h))
        self.motion_mask = thresh
        self.roi_regions = motion_regions
        return thresh, motion_regions
    
    def is_in_motion_region(self, bbox: Tuple[int, int, int, int], 
                            motion_regions: List[Tuple]) -> bool:
        x1, y1, x2, y2 = bbox
        for mx1, my1, mx2, my2 in motion_regions:
            if x1 < mx2 and x2 > mx1 and y1 < my2 and y2 > my1:
                return True
        return False


# ==================== LAYER 3: ADAPTIVE PROCESSING ====================

class AdaptiveProcessor:
    def __init__(self, target_fps: float = 60.0, min_scale: float = 0.5, 
                 max_scale: float = 1.0):
        self.target_fps = target_fps
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.current_scale = 1.0
        self.frame_skip = 0
        self.max_frame_skip = 2
        self.fps_history = deque(maxlen=30)
        self.skip_counter = 0
        self.processing_times = deque(maxlen=30)
        
    def get_adaptive_scale(self, current_fps: float) -> float:
        self.fps_history.append(current_fps)
        if len(self.fps_history) < 10:
            return self.current_scale
        avg_fps = np.mean(list(self.fps_history))
        fps_error = self.target_fps - avg_fps
        if fps_error < -5:
            self.current_scale = min(self.max_scale, self.current_scale + 0.05)
        elif fps_error > 5:
            self.current_scale = max(self.min_scale, self.current_scale - 0.05)
        if avg_fps < self.target_fps * 0.8:
            self.max_frame_skip = min(3, self.max_frame_skip + 1)
        elif avg_fps > self.target_fps * 0.95:
            self.max_frame_skip = max(0, self.max_frame_skip - 1)
        return self.current_scale
    
    def should_skip_frame(self) -> bool:
        if self.max_frame_skip == 0:
            return False
        self.skip_counter += 1
        if self.skip_counter > self.max_frame_skip:
            self.skip_counter = 0
            return False
        return self.skip_counter <= self.max_frame_skip - 1
    
    def resize_frame(self, frame: np.ndarray, scale: float = None) -> np.ndarray:
        if scale is None:
            scale = self.current_scale
        if scale >= 1.0:
            return frame
        new_width = int(frame.shape[1] * scale)
        new_height = int(frame.shape[0] * scale)
        return cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    
    def scale_bbox(self, bbox: Tuple, scale: float) -> Tuple:
        x1, y1, x2, y2 = bbox
        return (int(x1 / scale), int(y1 / scale), int(x2 / scale), int(y2 / scale))


# ==================== LAYER 4: GPU OPTIMIZATION ====================

class GPUOptimizer:
    def __init__(self, device: str = 'auto'):
        self.device = self._select_device(device)
        self.half_precision = self.device.type in ['cuda', 'mps']
        print(f"GPU Optimizer initialized on: {self.device}")
        if self.half_precision:
            print("Half precision (FP16) enabled")
    
    def _select_device(self, device: str) -> torch.device:
        if device == 'auto':
            if torch.cuda.is_available():
                return torch.device('cuda')
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return torch.device('mps')
            else:
                return torch.device('cpu')
        return torch.device(device)
    
    def optimize_model(self, model: YOLO) -> YOLO:
        model.to(self.device)
        return model


# ==================== LAYER 5: NMS OPTIMIZATION ====================

class NMSOptimizer:
    def __init__(self, iou_threshold: float = 0.5, score_threshold: float = 0.3,
                 use_soft_nms: bool = True):
        self.iou_threshold = iou_threshold
        self.score_threshold = score_threshold
        self.use_soft_nms = use_soft_nms
    
    def soft_nms(self, boxes: np.ndarray, scores: np.ndarray, 
                 sigma: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
        N = len(boxes)
        if N == 0:
            return np.array([]), np.array([])
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)
        keep = []
        new_scores = scores.copy()
        for i in range(N):
            max_idx = np.argmax(new_scores[i:]) + i
            boxes[[i, max_idx]] = boxes[[max_idx, i]]
            new_scores[[i, max_idx]] = new_scores[[max_idx, i]]
            areas[[i, max_idx]] = areas[[max_idx, i]]
            keep.append(i)
            if i == N - 1:
                break
            xx1 = np.maximum(boxes[i, 0], boxes[i+1:, 0])
            yy1 = np.maximum(boxes[i, 1], boxes[i+1:, 1])
            xx2 = np.minimum(boxes[i, 2], boxes[i+1:, 2])
            yy2 = np.minimum(boxes[i, 3], boxes[i+1:, 3])
            w = np.maximum(0, xx2 - xx1)
            h = np.maximum(0, yy2 - yy1)
            inter = w * h
            iou = inter / (areas[i] + areas[i+1:] - inter)
            new_scores[i+1:] *= np.exp(-(iou ** 2) / sigma)
        return np.array(keep), new_scores
    
    def class_aware_nms(self, detections: List[dict]) -> List[dict]:
        if not detections:
            return detections
        class_dets = defaultdict(list)
        for det in detections:
            class_dets[det['class']].append(det)
        results = []
        for class_name, dets in class_dets.items():
            boxes = np.array([d['bbox'] for d in dets])
            scores = np.array([d['confidence'] for d in dets])
            mask = scores >= self.score_threshold
            boxes = boxes[mask]
            scores = scores[mask]
            dets = [dets[i] for i in range(len(dets)) if mask[i]]
            if len(boxes) == 0:
                continue
            if self.use_soft_nms:
                keep_indices, _ = self.soft_nms(boxes, scores)
            else:
                keep_indices = cv2.dnn.NMSBoxes(
                    boxes.tolist(), scores.tolist(), 
                    self.score_threshold, self.iou_threshold
                )
                keep_indices = keep_indices.flatten() if len(keep_indices) > 0 else []
            for idx in keep_indices:
                if scores[idx] >= self.score_threshold:
                    results.append(dets[idx])
        return results


# ==================== LAYER 6: SPEED ESTIMATION ====================

class SpeedEstimator:
    def __init__(self, pixels_per_meter: float = 10.0, fps: float = 30.0):
        self.pixels_per_meter = 30.0
        self.fps = fps
        self.speed_smoothing = defaultdict(lambda: deque(maxlen=10))
        
    def estimate_speed(self, track_id: int, velocity: Tuple[float, float]) -> float:
        vx, vy = velocity
        speed_pixels = math.sqrt(vx**2 + vy**2)
        speed_ms = speed_pixels * self.fps / self.pixels_per_meter
        speed_kmh = speed_ms * 3.6
        self.speed_smoothing[track_id].append(speed_kmh)
        return np.mean(list(self.speed_smoothing[track_id]))
    
    def calibrate(self, pixel_distance: float, real_distance: float):
        self.pixels_per_meter = pixel_distance / real_distance


# ==================== STREAM LOADER ====================

class StreamLoader:
    def __init__(self, src, queue_size=128):
        self.src = src
        self.stream = cv2.VideoCapture(src)
        self.stopped = False
        self.queue_size = queue_size
        self.Q = deque(maxlen=queue_size)
        self.lock = threading.Lock()
        if not self.stream.isOpened():
            print(f"[ERROR] Failed to open stream: {src}")
            
    def start(self):
        t = threading.Thread(target=self.update, args=())
        t.daemon = True
        t.start()
        return self
    
    def update(self):
        while not self.stopped:
            if not self.stream.isOpened():
                self._reconnect()
                continue
            grabbed, frame = self.stream.read()
            if not grabbed:
                if isinstance(self.src, str) and (self.src.startswith('http') or self.src.startswith('rtsp')):
                    print(f"\n[WARNING] Stream read failed. Reconnecting...")
                    self._reconnect()
                else:
                    self.stop()
                continue
            with self.lock:
                self.Q.append(frame)
            if len(self.Q) >= self.queue_size:
                time.sleep(0.01)
            
    def _reconnect(self):
        if self.stream.isOpened():
            self.stream.release()
        time.sleep(1)
        self.stream = cv2.VideoCapture(self.src)
        if self.stream.isOpened():
            print("[INFO] Stream reconnected")
        else:
            print("[WARNING] Reconnection failed")
            
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

# ==================== MAIN TRAFFIC MONITOR CLASS ====================

class TrafficMonitorKalmanOptimized:
    def __init__(self, model_path: str = 'yolov8n.pt', conf_threshold: float = 0.4,
                 target_fps: float = 60.0, enable_motion_filter: bool = True):
        print("=" * 70)
        print("INITIALIZING TRAFFIC MONITORING SYSTEM - KALMAN FILTER OPTIMIZED")
        print("=" * 70)
        print(f"Target FPS: {target_fps}")
        
        self.vehicle_classes = {2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'}
        self.colors = {
            'car': (0, 255, 0),
            'motorcycle': (0, 0, 255),
            'bus': (255, 0, 0),
            'truck': (255, 255, 0)
        }
        
        self.gpu_optimizer = GPUOptimizer()
        
        print(f"\nLoading model: {model_path}")
        self.model = YOLO(model_path)
        self.model = self.gpu_optimizer.optimize_model(self.model)
        self.conf_threshold = conf_threshold
        
        self.kalman_tracker = KalmanTrackerManager(
            max_age=30, min_hits=2, iou_threshold=0.3, distance_threshold=100
        )
        
        self.enable_motion_filter = enable_motion_filter
        self.motion_detector = MotionDetector() if enable_motion_filter else None
        
        self.adaptive_processor = AdaptiveProcessor(target_fps=target_fps)
        
        self.nms_optimizer = NMSOptimizer(
            iou_threshold=0.5, score_threshold=conf_threshold, use_soft_nms=True
        )
        
        self.speed_estimator = SpeedEstimator()
        
        self.vehicle_count = defaultdict(int)
        self.total_count = 0
        self.counted_ids = set()
        self.line_position = None
        self.fps = 0
        self.fps_history = deque(maxlen=30)
        self.prev_frame_time = 0
        self.density_history = deque(maxlen=300)
        self.congestion_level = "normal"
        self.log_data = []
        self.speed_limit = 60
        
        print("\nAll layers initialized successfully!")
        print("=" * 70)
    
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, List[dict], dict]:
        speed = 0.0
        current_time = time.time()
        if self.prev_frame_time > 0:
            dt = current_time - self.prev_frame_time
            self.fps = 1 / dt if dt > 0 else 0
        self.prev_frame_time = current_time
        self.fps_history.append(self.fps)
        
        scale = self.adaptive_processor.get_adaptive_scale(self.fps)
        
        if self.adaptive_processor.should_skip_frame():
            result_stats = {
                'fps': self.fps, 'skipped': True,
                'total_count': self.total_count, 'congestion': "N/A"
            }
            if hasattr(self, 'last_annotated_frame') and self.last_annotated_frame is not None:
                return self.last_annotated_frame, [], result_stats
            else:
                return frame, [], result_stats
        
        processed_frame = self.adaptive_processor.resize_frame(frame, scale)
        
        if self.enable_motion_filter and self.motion_detector:
            _, motion_regions = self.motion_detector.detect_motion(processed_frame)
        
        results = self.model.track(
            processed_frame, conf=self.conf_threshold,
            persist=True, verbose=False, iou=0.5
        )
        
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls_id = int(box.cls[0])
                if cls_id in self.vehicle_classes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    class_name = self.vehicle_classes[cls_id]
                    if scale != 1.0:
                        x1, y1, x2, y2 = self.adaptive_processor.scale_bbox((x1, y1, x2, y2), scale)
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2
                    detections.append({
                        'class': class_name, 'confidence': conf,
                        'bbox': (x1, y1, x2, y2), 'center': (center_x, center_y)
                    })
        
        detections = self.nms_optimizer.class_aware_nms(detections)
        tracks = self.kalman_tracker.update(detections)
        
        self.speed_estimator.fps = max(self.fps, 30)
        if self.speed_estimator.pixels_per_meter == 10.0:
            self.speed_estimator.pixels_per_meter = frame.shape[0] / 15.0
        
        if self.line_position is None:
            self.line_position = frame.shape[0] // 2
        
        annotated_frame = frame.copy()
        cv2.line(annotated_frame, (0, self.line_position),
                (frame.shape[1], self.line_position), (0, 255, 255), 2)
        
        speed = 0.0
        for track in tracks:
            track_id = track['track_id']
            class_name = track['class']
            bbox = track['bbox']
            center = track['center']
            confidence = track['confidence']
            velocity = track.get('velocity', (0, 0))
            history = track.get('history', [])
            
            speed = self.speed_estimator.estimate_speed(track_id, velocity)
            
            if center[1] > self.line_position - 20 and center[1] < self.line_position + 20:
                if track_id not in self.counted_ids:
                    self.counted_ids.add(track_id)
                    self.vehicle_count[class_name] += 1
                    self.total_count += 1
            
            color = self.colors.get(class_name, (255, 255, 255))
            x1, y1, x2, y2 = bbox
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            
            if len(history) > 1:
                for i in range(1, len(history)):
                    cv2.line(annotated_frame, history[i-1], history[i], color, 1)
            
            cv2.circle(annotated_frame, center, 4, (0, 0, 255), -1)
            vx, vy = velocity
            end_point = (int(center[0] + vx * 5), int(center[1] + vy * 5))
            cv2.arrowedLine(annotated_frame, center, end_point, (255, 0, 255), 2)
            
            label = f"{class_name} #{track_id} {speed:.0f}km/h"
            (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(annotated_frame, (x1, y1 - text_h - 8), (x1 + text_w, y1), color, -1)
            cv2.putText(annotated_frame, label, (x1, y1 - 4),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        density = self.calculate_density(len(tracks), frame.shape)
        congestion, congestion_color = self.detect_congestion(density)
        self._draw_info_panel(annotated_frame, speed, congestion, congestion_color)
        
        stats = {
            'fps': self.fps, 'target_fps': self.adaptive_processor.target_fps,
            'scale': scale, 'total_count': self.total_count,
            'vehicle_count': dict(self.vehicle_count), 'active_tracks': len(tracks),
            'density': density, 'congestion': congestion
        }
        self.last_annotated_frame = annotated_frame
        return annotated_frame, tracks, stats
    
    def calculate_density(self, num_vehicles: int, frame_shape: Tuple[int, int]) -> float:
        area = frame_shape[0] * frame_shape[1]
        avg_vehicle_area = 30 * 15
        density = (num_vehicles * avg_vehicle_area) / area
        self.density_history.append(density)
        return density
    
    def detect_congestion(self, density: float) -> Tuple[str, str]:
        if density < 0.05:
            self.congestion_level = "LANCAR"
            color = "green"
        elif density < 0.1:
            self.congestion_level = "RAMAI"
            color = "yellow"
        elif density < 0.2:
            self.congestion_level = "PADAT"
            color = "orange"
        else:
            self.congestion_level = "MACET"
            color = "red"
        return self.congestion_level, color
    
    def _draw_info_panel(self, frame: np.ndarray, avg_speed: float, 
                         congestion: str, congestion_color: str):
        cv2.rectangle(frame, (10, 10), (350, 180), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (350, 180), (0, 255, 255), 2)
        cv2.putText(frame, "ATCS TRACKING - KALMAN FILTER", (20, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        avg_fps = np.mean(list(self.fps_history)) if self.fps_history else self.fps
        fps_color = (0, 255, 0) if avg_fps >= 60 else (0, 255, 255) if avg_fps >= 30 else (0, 0, 255)
        cv2.putText(frame, f"FPS: {avg_fps:.1f}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, fps_color, 1)
        cv2.putText(frame, f"Total Vehicles: {self.total_count}", (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y_offset = 125
        for vtype, count in self.vehicle_count.items():
            color = self.colors.get(vtype, (255, 255, 255))
            cv2.putText(frame, f"{vtype.capitalize()}: {count}", 
                       (20 + (y_offset - 125) * 110, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            if (y_offset - 125) % 2 == 1:
                y_offset += 20
        color_map = {'green': (0, 255, 0), 'yellow': (0, 255, 255), 
                    'orange': (0, 165, 255), 'red': (0, 0, 255)}
        cv2.putText(frame, f"Status: {congestion}", (20, 165),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_map.get(congestion_color, (255, 255, 255)), 2)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(frame, timestamp, (frame.shape[1] - 250, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    def process_video(self, source, output_path: str = None, show_display: bool = True):
        print(f"Initializing threaded stream loader for: {source}")
        stream_loader = StreamLoader(source).start()
        time.sleep(2.0)
        
        fps = stream_loader.stream.get(cv2.CAP_PROP_FPS) or 30
        width = int(stream_loader.stream.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(stream_loader.stream.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.speed_estimator.fps = fps
        
        print(f"\nConnected to ATCS: {width}x{height} @ {fps}fps")
        
        frame_count = 0
        start_time = time.time()
        
        try:
            while True:
                if not stream_loader.running():
                    break
                ret, frame = stream_loader.read()
                if not ret:
                    time.sleep(0.01)
                    continue
                
                processed_frame, tracks, stats = self.process_frame(frame)
                
                if show_display:
                    cv2.imshow('ATCS MONITOR - Direct Link', processed_frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    elif key == ord('s'):
                        screenshot = f"screenshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                        cv2.imwrite(screenshot, processed_frame)
                        print(f"Screenshot saved: {screenshot}")
                
                frame_count += 1
                if frame_count % 30 == 0:
                    print(f"FPS: {stats['fps']:.1f} | Total: {stats['total_count']} | Status: {stats['congestion']}")
        
        except KeyboardInterrupt:
            print("\nStopped by user")
        finally:
            stream_loader.stop()
            cv2.destroyAllWindows()
            self._print_summary(frame_count, start_time)
    
    def _print_summary(self, frame_count: int, start_time: float):
        elapsed = time.time() - start_time
        print("\n" + "=" * 70)
        print("SESSION SUMMARY")
        print("=" * 70)
        print(f"Duration: {elapsed:.1f}s")
        print(f"Total Vehicles: {self.total_count}")
        print("=" * 70)


# ==================== MAIN ENTRY POINT ====================

def main():
    print("=" * 70)
    print("ATCS DIRECT MONITORING SYSTEM")
    print("Target: Connecting to Wirosaban/Custom Link...")
    print("=" * 70)
    
    # Initialize without asking user questions
    monitor = TrafficMonitorKalmanOptimized(
        model_path=MODEL_FILE,
        conf_threshold=0.4,
        target_fps=TARGET_FPS,
        enable_motion_filter=True
    )
    
    print(f"\nMemulai deteksi stream ATCS...")
    print(f"URL: {ATCS_URL}")
    print("Tekan 'q' untuk berhenti, 's' untuk screenshot\n")
    
    # Start process immediately with hardcoded URL
    monitor.process_video(
        ATCS_URL,
        output_path=None, # Disable recording for performance
        show_display=True
    )

if __name__ == "__main__":
    main()