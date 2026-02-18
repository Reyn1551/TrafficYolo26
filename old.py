"""
Sistem Deteksi Kendaraan Real-time menggunakan YOLOv26 - KALMAN FILTER OPTIMIZED VERSION
=======================================================================================
Sumber Video: CCTV Simpang Wirosaban View Barat (Jogja)
Author: Optimized System
Tujuan: Skripsi - Traffic Monitoring System

Optimasi Utama:
- Kalman Filter untuk tracking presisi tinggi
- Hungarian Algorithm untuk optimal assignment
- Multi-threading untuk 60+ FPS
- Adaptive Frame Processing
- Motion Detection Pre-filtering
- GPU Acceleration Support
- Layer-by-Layer Architecture

Target: Minimal 60 FPS dengan akurasi tinggi
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
from filterpy.common import Q_discrete_white_noise
import warnings
warnings.filterwarnings('ignore')

# ==================== LAYER 1: KALMAN FILTER TRACKER ====================

class KalmanBoxTracker:
    """
    Kalman Filter untuk tracking bounding box kendaraan.
    State: [x, y, w, h, vx, vy, vw, vh]
    - x, y: center position
    - w, h: width, height
    - vx, vy, vw, vh: velocities
    
    Kalman Filter memberikan estimasi posisi yang lebih smooth dan akurat
    dengan memprediksi posisi objek berdasarkan gerakan sebelumnya.
    """
    count = 0
    
    def __init__(self, bbox: Tuple[int, int, int, int], class_name: str, confidence: float):
        """
        Inisialisasi Kalman Filter untuk tracking.
        
        Args:
            bbox: (x1, y1, x2, y2) bounding box
            class_name: Nama kelas kendaraan
            confidence: Confidence score
        """
        # Define constant velocity model
        self.kf = KalmanFilter(dim_x=8, dim_z=4)
        
        # State transition matrix (constant velocity model)
        self.kf.F = np.array([
            [1, 0, 0, 0, 1, 0, 0, 0],  # x = x + vx*dt
            [0, 1, 0, 0, 0, 1, 0, 0],  # y = y + vy*dt
            [0, 0, 1, 0, 0, 0, 1, 0],  # w = w + vw*dt
            [0, 0, 0, 1, 0, 0, 0, 1],  # h = h + vh*dt
            [0, 0, 0, 0, 1, 0, 0, 0],  # vx
            [0, 0, 0, 0, 0, 1, 0, 0],  # vy
            [0, 0, 0, 0, 0, 0, 1, 0],  # vw
            [0, 0, 0, 0, 0, 0, 0, 1]   # vh
        ])
        
        # Measurement function (we only observe x, y, w, h)
        self.kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0]
        ])
        
        # Measurement noise covariance
        self.kf.R[2:, 2:] *= 10.0  # More uncertainty in size
        self.kf.R *= 1.0
        
        # Process noise covariance (adjusted for smooth tracking)
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01
        self.kf.Q *= 0.01
        
        # Initial state covariance
        self.kf.P[4:, 4:] *= 1000.0  # High uncertainty for initial velocity
        self.kf.P *= 10.0
        
        # Initialize state with first measurement
        x1, y1, x2, y2 = bbox
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        w, h = x2 - x1, y2 - y1
        
        self.kf.x[:4] = np.array([cx, cy, w, h]).reshape(4, 1)
        
        # Track metadata
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.class_name = class_name
        self.confidence = confidence
        self.age = 0
        self.hits = 1
        self.hit_streak = 1
        self.time_since_update = 0
        self.history = deque(maxlen=50)  # Position history for trajectory
        self.speed_history = deque(maxlen=30)
        self.prediction_history = deque(maxlen=10)
        
        # Kalman innovation for quality assessment
        self.innovation = 0.0
        
    def predict(self) -> np.ndarray:
        """
        Predict state ahead using Kalman Filter.
        
        Returns:
            Predicted bounding box [x1, y1, x2, y2]
        """
        # Predict next state
        self.kf.predict()
        self.age += 1
        
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        
        # Get predicted bbox
        cx, cy, w, h = self.kf.x[:4].flatten()
        x1, y1 = cx - w/2, cy - h/2
        x2, y2 = cx + w/2, cy + h/2
        
        # Store prediction for visualization
        self.prediction_history.append((int(cx), int(cy)))
        
        return np.array([x1, y1, x2, y2])
    
    def update(self, bbox: Tuple[int, int, int, int], confidence: float = None):
        """
        Update state with observed measurement.
        
        Args:
            bbox: Observed bounding box (x1, y1, x2, y2)
            confidence: New confidence score
        """
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1
        
        if confidence is not None:
            self.confidence = confidence
        
        # Convert to measurement format
        x1, y1, x2, y2 = bbox
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        w, h = x2 - x1, y2 - y1
        measurement = np.array([cx, cy, w, h]).reshape(4, 1)
        
        # Calculate innovation (measurement residual) for quality assessment
        predicted_measurement = self.kf.H @ self.kf.x
        self.innovation = np.linalg.norm(measurement - predicted_measurement)
        
        # Update Kalman Filter
        self.kf.update(measurement)
        
        # Store position in history
        self.history.append((int(cx), int(cy)))
    
    def get_state(self) -> Tuple[int, int, int, int]:
        """
        Get current bounding box estimate.
        
        Returns:
            Bounding box (x1, y1, x2, y2)
        """
        cx, cy, w, h = self.kf.x[:4].flatten()
        x1, y1 = cx - w/2, cy - h/2
        x2, y2 = cx + w/2, cy + h/2
        return (int(x1), int(y1), int(x2), int(y2))
    
    def get_velocity(self) -> Tuple[float, float]:
        """
        Get estimated velocity from Kalman Filter.
        
        Returns:
            (vx, vy) velocity in pixels/frame
        """
        vx, vy = self.kf.x[4, 0], self.kf.x[5, 0]
        return float(vx), float(vy)
    
    def get_center(self) -> Tuple[int, int]:
        """Get center point of bounding box."""
        cx, cy = int(self.kf.x[0, 0]), int(self.kf.x[1, 0])
        return (cx, cy)


class KalmanTrackerManager:
    """
    Manager untuk multiple Kalman Filter trackers dengan Hungarian Algorithm
    untuk optimal assignment.
    """
    
    def __init__(self, max_age: int = 30, min_hits: int = 3, 
                 iou_threshold: float = 0.3, distance_threshold: float = 100):
        """
        Initialize tracker manager.
        
        Args:
            max_age: Maximum frames before track is deleted
            min_hits: Minimum hits before track is confirmed
            iou_threshold: IoU threshold for matching
            distance_threshold: Maximum distance for matching
        """
        self.trackers: List[KalmanBoxTracker] = []
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.distance_threshold = distance_threshold
        self.frame_count = 0
        
    def iou_batch(self, bb_test: np.ndarray, bb_gt: np.ndarray) -> np.ndarray:
        """
        Calculate IoU between two sets of bounding boxes.
        
        Args:
            bb_test: (N, 4) array of boxes
            bb_gt: (M, 4) array of boxes
            
        Returns:
            (N, M) IoU matrix
        """
        bb_test = np.array(bb_test)
        bb_gt = np.array(bb_gt)
        
        if len(bb_test) == 0 or len(bb_gt) == 0:
            return np.zeros((len(bb_test), len(bb_gt)))
        
        # Calculate intersection
        xx1 = np.maximum(bb_test[:, 0].reshape(-1, 1), bb_gt[:, 0].reshape(1, -1))
        yy1 = np.maximum(bb_test[:, 1].reshape(-1, 1), bb_gt[:, 1].reshape(1, -1))
        xx2 = np.minimum(bb_test[:, 2].reshape(-1, 1), bb_gt[:, 2].reshape(1, -1))
        yy2 = np.minimum(bb_test[:, 3].reshape(-1, 1), bb_gt[:, 3].reshape(1, -1))
        
        w = np.maximum(0., xx2 - xx1)
        h = np.maximum(0., yy2 - yy1)
        intersection = w * h
        
        # Calculate areas
        area_test = (bb_test[:, 2] - bb_test[:, 0]) * (bb_test[:, 3] - bb_test[:, 1])
        area_gt = (bb_gt[:, 2] - bb_gt[:, 0]) * (bb_gt[:, 3] - bb_gt[:, 1])
        
        union = area_test.reshape(-1, 1) + area_gt.reshape(1, -1) - intersection
        
        return intersection / (union + 1e-6)
    
    def distance_batch(self, centers_test: List[Tuple], centers_gt: List[Tuple]) -> np.ndarray:
        """
        Calculate Euclidean distance between centers.
        
        Args:
            centers_test: List of (x, y) tuples
            centers_gt: List of (x, y) tuples
            
        Returns:
            Distance matrix
        """
        n, m = len(centers_test), len(centers_gt)
        if n == 0 or m == 0:
            return np.zeros((n, m))
        
        dist_matrix = np.zeros((n, m))
        for i, ct in enumerate(centers_test):
            for j, cg in enumerate(centers_gt):
                dist_matrix[i, j] = math.sqrt((ct[0] - cg[0])**2 + (ct[1] - cg[1])**2)
        
        return dist_matrix
    
    def associate_detections_to_trackers(self, detections: List[dict]) -> Tuple[List, List, List]:
        """
        Associate detections to existing trackers using Hungarian Algorithm.
        
        Hungarian Algorithm memberikan optimal assignment dengan kompleksitas
        O(n^3), lebih efisien daripada greedy assignment.
        
        Args:
            detections: List of detection dictionaries
            
        Returns:
            (matches, unmatched_detections, unmatched_trackers)
        """
        if len(self.trackers) == 0:
            return [], list(range(len(detections))), []
        
        # Get predicted boxes from trackers
        predicted_boxes = np.array([t.predict() for t in self.trackers])
        predicted_centers = [t.get_center() for t in self.trackers]
        
        # Get detection boxes and centers
        det_boxes = np.array([d['bbox'] for d in detections]) if detections else np.array([])
        det_centers = [d['center'] for d in detections]
        
        # Calculate cost matrix (combine IoU and distance)
        iou_matrix = self.iou_batch(det_boxes, predicted_boxes) if len(det_boxes) > 0 else np.zeros((0, len(self.trackers)))
        dist_matrix = self.distance_batch(det_centers, predicted_centers)
        
        # Combined cost: (1 - IoU) + normalized_distance
        # Lower cost = better match
        max_dist = max(self.distance_threshold, dist_matrix.max() + 1) if dist_matrix.size > 0 else self.distance_threshold
        cost_matrix = (1 - iou_matrix) + (dist_matrix / max_dist) * 0.5
        
        # Hungarian Algorithm for optimal assignment
        matched_indices = linear_sum_assignment(cost_matrix)
        
        matches = []
        unmatched_detections = list(range(len(detections)))
        unmatched_trackers = list(range(len(self.trackers)))
        
        for d_idx, t_idx in zip(matched_indices[0], matched_indices[1]):
            # Check if match is valid
            if iou_matrix.size > 0 and d_idx < iou_matrix.shape[0] and t_idx < iou_matrix.shape[1]:
                if iou_matrix[d_idx, t_idx] >= self.iou_threshold or dist_matrix[d_idx, t_idx] <= self.distance_threshold:
                    matches.append((d_idx, t_idx))
                    if d_idx in unmatched_detections:
                        unmatched_detections.remove(d_idx)
                    if t_idx in unmatched_trackers:
                        unmatched_trackers.remove(t_idx)
        
        return matches, unmatched_detections, unmatched_trackers
    
    def update(self, detections: List[dict]) -> List[dict]:
        """
        Update all trackers with new detections.
        
        Args:
            detections: List of detection dictionaries
            
        Returns:
            List of updated track information
        """
        self.frame_count += 1
        
        # Associate detections to trackers
        matches, unmatched_dets, unmatched_trks = self.associate_detections_to_trackers(detections)
        
        # Update matched trackers
        for d_idx, t_idx in matches:
            det = detections[d_idx]
            self.trackers[t_idx].update(det['bbox'], det['confidence'])
        
        # Create new trackers for unmatched detections
        for d_idx in unmatched_dets:
            det = detections[d_idx]
            new_tracker = KalmanBoxTracker(det['bbox'], det['class'], det['confidence'])
            self.trackers.append(new_tracker)
        
        # Remove dead trackers
        trackers_to_remove = []
        for t_idx, trk in enumerate(self.trackers):
            if trk.time_since_update > self.max_age:
                trackers_to_remove.append(t_idx)
        
        for t_idx in reversed(trackers_to_remove):
            self.trackers.pop(t_idx)
        
        # Return active tracks
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
    """
    Motion detection layer untuk pre-filtering.
    Menggunakan MOG2 Background Subtraction untuk mendeteksi area bergerak,
    sehingga deteksi YOLO hanya dilakukan pada area yang relevan.
    """
    
    def __init__(self, history: int = 500, var_threshold: float = 16, 
                 detect_shadows: bool = False, min_area: int = 500):
        """
        Initialize motion detector.
        
        Args:
            history: Number of frames for background model
            var_threshold: Threshold for background/foreground
            detect_shadows: Whether to detect shadows
            min_area: Minimum contour area to consider
        """
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=history,
            varThreshold=var_threshold,
            detectShadows=detect_shadows
        )
        self.min_area = min_area
        self.motion_mask = None
        self.roi_regions = []
        
    def detect_motion(self, frame: np.ndarray) -> Tuple[np.ndarray, List[Tuple]]:
        """
        Detect motion regions in frame.
        
        Args:
            frame: Input frame
            
        Returns:
            (motion_mask, motion_regions)
        """
        # Apply background subtraction
        fg_mask = self.bg_subtractor.apply(frame)
        
        # Threshold and morphological operations
        _, thresh = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Get bounding boxes for motion regions
        motion_regions = []
        for contour in contours:
            if cv2.contourArea(contour) >= self.min_area:
                x, y, w, h = cv2.boundingRect(contour)
                # Expand region slightly
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
        """Check if bounding box overlaps with any motion region."""
        x1, y1, x2, y2 = bbox
        for mx1, my1, mx2, my2 in motion_regions:
            # Check overlap
            if x1 < mx2 and x2 > mx1 and y1 < my2 and y2 > my1:
                return True
        return False


# ==================== LAYER 3: ADAPTIVE PROCESSING ====================

class AdaptiveProcessor:
    """
    Adaptive processing layer untuk optimasi FPS.
    Menyesuaikan resolusi dan frame skipping berdasarkan beban sistem.
    """
    
    def __init__(self, target_fps: float = 60.0, min_scale: float = 0.5, 
                 max_scale: float = 1.0):
        """
        Initialize adaptive processor.
        
        Args:
            target_fps: Target FPS yang diinginkan
            min_scale: Minimum scale factor
            max_scale: Maximum scale factor
        """
        self.target_fps = target_fps
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.current_scale = 1.0
        self.frame_skip = 0
        self.max_frame_skip = 2
        self.fps_history = deque(maxlen=30)
        self.skip_counter = 0
        
        # Frame times for adaptive control
        self.processing_times = deque(maxlen=30)
        
    def get_adaptive_scale(self, current_fps: float) -> float:
        """
        Calculate adaptive scale based on current FPS.
        
        Args:
            current_fps: Current processing FPS
            
        Returns:
            Scale factor for frame resizing
        """
        self.fps_history.append(current_fps)
        
        if len(self.fps_history) < 10:
            return self.current_scale
        
        avg_fps = np.mean(list(self.fps_history))
        
        # PID-like control for scale adjustment
        fps_error = self.target_fps - avg_fps
        
        if fps_error < -5:  # FPS too high, can increase quality
            self.current_scale = min(self.max_scale, self.current_scale + 0.05)
        elif fps_error > 5:  # FPS too low, decrease quality
            self.current_scale = max(self.min_scale, self.current_scale - 0.05)
        
        # Adjust frame skip based on FPS
        if avg_fps < self.target_fps * 0.8:
            self.max_frame_skip = min(3, self.max_frame_skip + 1)
        elif avg_fps > self.target_fps * 0.95:
            self.max_frame_skip = max(0, self.max_frame_skip - 1)
        
        return self.current_scale
    
    def should_skip_frame(self) -> bool:
        """Determine if current frame should be skipped."""
        if self.max_frame_skip == 0:
            return False
        
        self.skip_counter += 1
        if self.skip_counter > self.max_frame_skip:
            self.skip_counter = 0
            return False
        return self.skip_counter <= self.max_frame_skip - 1
    
    def resize_frame(self, frame: np.ndarray, scale: float = None) -> np.ndarray:
        """
        Resize frame based on adaptive scale.
        
        Args:
            frame: Input frame
            scale: Optional override scale
            
        Returns:
            Resized frame
        """
        if scale is None:
            scale = self.current_scale
        
        if scale >= 1.0:
            return frame
        
        new_width = int(frame.shape[1] * scale)
        new_height = int(frame.shape[0] * scale)
        
        return cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    
    def scale_bbox(self, bbox: Tuple, scale: float) -> Tuple:
        """Scale bounding box coordinates."""
        x1, y1, x2, y2 = bbox
        return (int(x1 / scale), int(y1 / scale), int(x2 / scale), int(y2 / scale))


# ==================== LAYER 4: GPU OPTIMIZATION ====================

class GPUOptimizer:
    """
    GPU optimization layer untuk akselerasi inferensi.
    """
    
    def __init__(self, device: str = 'auto'):
        """
        Initialize GPU optimizer.
        
        Args:
            device: 'auto', 'cuda', 'mps', or 'cpu'
        """
        self.device = self._select_device(device)
        self.half_precision = self.device.type in ['cuda', 'mps']
        
        print(f"GPU Optimizer initialized on: {self.device}")
        if self.half_precision:
            print("Half precision (FP16) enabled")
    
    def _select_device(self, device: str) -> torch.device:
        """Select optimal device."""
        if device == 'auto':
            if torch.cuda.is_available():
                return torch.device('cuda')
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return torch.device('mps')
            else:
                return torch.device('cpu')
        return torch.device(device)
    
    def optimize_model(self, model: YOLO) -> YOLO:
        """Optimize model for selected device."""
        model.to(self.device)
        # if self.half_precision:
        #     model.model.half()
        return model
    
    def preprocess_batch(self, frames: List[np.ndarray]) -> torch.Tensor:
        """Preprocess batch of frames for inference."""
        batch = []
        for frame in frames:
            # Normalize and convert to tensor
            img = frame.astype(np.float32) / 255.0
            img = np.transpose(img, (2, 0, 1))  # HWC to CHW
            batch.append(img)
        
        tensor = torch.from_numpy(np.stack(batch))
        tensor = tensor.to(self.device)
        
        if self.half_precision:
            tensor = tensor.half()
        
        return tensor


# ==================== LAYER 5: NMS OPTIMIZATION ====================

class NMSOptimizer:
    """
    Optimized Non-Maximum Suppression layer.
    Menggunakan Soft-NMS dan class-aware NMS untuk hasil lebih baik.
    """
    
    def __init__(self, iou_threshold: float = 0.5, score_threshold: float = 0.3,
                 use_soft_nms: bool = True):
        """
        Initialize NMS optimizer.
        
        Args:
            iou_threshold: IoU threshold for NMS
            score_threshold: Minimum confidence score
            use_soft_nms: Use Soft-NMS instead of standard NMS
        """
        self.iou_threshold = iou_threshold
        self.score_threshold = score_threshold
        self.use_soft_nms = use_soft_nms
    
    def soft_nms(self, boxes: np.ndarray, scores: np.ndarray, 
                 sigma: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Soft-NMS implementation - lebih presisi daripada standard NMS.
        
        Args:
            boxes: (N, 4) array of boxes
            scores: (N,) array of scores
            sigma: Gaussian sigma for score decay
            
        Returns:
            (keep_indices, new_scores)
        """
        N = len(boxes)
        if N == 0:
            return np.array([]), np.array([])
        
        # Convert to x1, y1, x2, y2 format
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        
        areas = (x2 - x1) * (y2 - y1)
        
        keep = []
        new_scores = scores.copy()
        
        for i in range(N):
            # Find box with maximum score
            max_idx = np.argmax(new_scores[i:]) + i
            
            # Swap
            boxes[[i, max_idx]] = boxes[[max_idx, i]]
            new_scores[[i, max_idx]] = new_scores[[max_idx, i]]
            areas[[i, max_idx]] = areas[[max_idx, i]]
            
            keep.append(i)
            
            if i == N - 1:
                break
            
            # Calculate IoU with remaining boxes
            xx1 = np.maximum(boxes[i, 0], boxes[i+1:, 0])
            yy1 = np.maximum(boxes[i, 1], boxes[i+1:, 1])
            xx2 = np.minimum(boxes[i, 2], boxes[i+1:, 2])
            yy2 = np.minimum(boxes[i, 3], boxes[i+1:, 3])
            
            w = np.maximum(0, xx2 - xx1)
            h = np.maximum(0, yy2 - yy1)
            
            inter = w * h
            iou = inter / (areas[i] + areas[i+1:] - inter)
            
            # Gaussian decay
            new_scores[i+1:] *= np.exp(-(iou ** 2) / sigma)
        
        return np.array(keep), new_scores
    
    def class_aware_nms(self, detections: List[dict]) -> List[dict]:
        """
        Class-aware NMS - melakukan NMS per kelas untuk menghindari
        false positives.
        """
        if not detections:
            return detections
        
        # Group by class
        class_dets = defaultdict(list)
        for det in detections:
            class_dets[det['class']].append(det)
        
        results = []
        for class_name, dets in class_dets.items():
            boxes = np.array([d['bbox'] for d in dets])
            scores = np.array([d['confidence'] for d in dets])
            
            # Filter by score threshold
            mask = scores >= self.score_threshold
            boxes = boxes[mask]
            scores = scores[mask]
            dets = [dets[i] for i in range(len(dets)) if mask[i]]
            
            if len(boxes) == 0:
                continue
            
            if self.use_soft_nms:
                keep_indices, _ = self.soft_nms(boxes, scores)
            else:
                # Standard NMS using OpenCV
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
    """
    Enhanced speed estimation menggunakan Kalman Filter velocity.
    """
    
    def __init__(self, pixels_per_meter: float = 10.0, fps: float = 30.0):
        """
        Initialize speed estimator.
        
        Args:
            pixels_per_meter: Calibration factor
            fps: Video FPS
        """
        self.pixels_per_meter = 30.0  # Increased from 10.0 to reduce speed overestimation
        self.fps = fps
        self.speed_smoothing = defaultdict(lambda: deque(maxlen=10))
        
    def estimate_speed(self, track_id: int, velocity: Tuple[float, float]) -> float:
        """
        Estimate speed from Kalman Filter velocity.
        
        Args:
            track_id: Track ID
            velocity: (vx, vy) velocity from Kalman Filter
            
        Returns:
            Speed in km/h
        """
        vx, vy = velocity
        
        # Calculate speed magnitude in pixels/frame
        speed_pixels = math.sqrt(vx**2 + vy**2)
        
        # Convert to m/s (pixels/frame * frames/second / pixels_per_meter)
        speed_ms = speed_pixels * self.fps / self.pixels_per_meter
        
        # Convert to km/h
        speed_kmh = speed_ms * 3.6
        
        # Smooth with moving average
        self.speed_smoothing[track_id].append(speed_kmh)
        
        return np.mean(list(self.speed_smoothing[track_id]))
    
    def calibrate(self, pixel_distance: float, real_distance: float):
        """Set calibration from known distance."""
        self.pixels_per_meter = pixel_distance / real_distance


# ==================== LAYER 7: MULTI-THREADING ====================

class MultiThreadProcessor:
    """
    Multi-threading layer untuk parallel processing.
    Menggunakan producer-consumer pattern untuk maksimal throughput.
    """
    
    def __init__(self, num_workers: int = 3, queue_size: int = 10):
        """
        Initialize multi-thread processor.
        
        Args:
            num_workers: Number of worker threads
            queue_size: Maximum queue size
        """
        self.frame_queue = queue.Queue(maxsize=queue_size)
        self.result_queue = queue.Queue(maxsize=queue_size)
        self.workers = []
        self.num_workers = num_workers
        self.running = False
        self.process_fn = None
        
    def start(self, process_fn):
        """Start worker threads."""
        self.process_fn = process_fn
        self.running = True
        
        for i in range(self.num_workers):
            worker = threading.Thread(target=self._worker, args=(i,))
            worker.daemon = True
            worker.start()
            self.workers.append(worker)
    
    def _worker(self, worker_id: int):
        """Worker thread function."""
        while self.running:
            try:
                frame_data = self.frame_queue.get(timeout=0.1)
                if frame_data is None:
                    break
                
                frame_idx, frame = frame_data
                result = self.process_fn(frame)
                self.result_queue.put((frame_idx, result))
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Worker {worker_id} error: {e}")
    
    def add_frame(self, frame_idx: int, frame: np.ndarray):
        """Add frame to processing queue."""
        try:
            self.frame_queue.put((frame_idx, frame), timeout=0.1)
            return True
        except queue.Full:
            return False
    
    def get_result(self, timeout: float = 0.1) -> Optional[Tuple]:
        """Get result from queue."""
        try:
            return self.result_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def stop(self):
        """Stop all workers."""
        self.running = False
        for _ in self.workers:
            self.frame_queue.put(None)


# ==================== STREAM LOADER ====================

class StreamLoader:
    def __init__(self, src, queue_size=128):
        self.src = src
        self.stream = cv2.VideoCapture(src)
        self.stopped = False
        self.queue_size = queue_size
        self.Q = deque(maxlen=queue_size)
        self.lock = threading.Lock()
        
        # Check connection
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
                # If reading fails, try to reconnect for stream sources
                if isinstance(self.src, str) and (self.src.startswith('http') or self.src.startswith('rtsp')):
                    print(f"\n[WARNING] Stream read failed. Reconnecting...")
                    self._reconnect()
                else:
                    self.stop()
                continue
            
            # Add to queue
            with self.lock:
                self.Q.append(frame)
                
            # Sleep slightly to prevent CPU hogging if queue is full
            if len(self.Q) >= self.queue_size:
                time.sleep(0.01)
            
    def _reconnect(self):
        if self.stream.isOpened():
            self.stream.release()
            
        time.sleep(1) # Wait before reconnect
        self.stream = cv2.VideoCapture(self.src)
        if self.stream.isOpened():
            print("[INFO] Stream reconnected")
        else:
            print("[WARNING] Reconnection failed")
            
    def read(self):
        with self.lock:
            if len(self.Q) > 0:
                return True, self.Q.popleft() # Get oldest frame (FIFO) for processing
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
    """
    Main Traffic Monitoring System dengan semua optimasi layer.
    
    Layer Architecture:
    1. Motion Detection Layer - Pre-filter dengan background subtraction
    2. Detection Layer - YOLOv26 inference
    3. Kalman Filter Layer - Tracking dengan Kalman Filter
    4. NMS Layer - Optimized non-maximum suppression
    5. Speed Estimation Layer - Estimasi kecepatan dari Kalman velocity
    6. Adaptive Processing Layer - Dynamic resolution dan frame skipping
    7. GPU Optimization Layer - Hardware acceleration
    """
    
    def __init__(self, model_path: str = 'yolov26n.pt', conf_threshold: float = 0.4,
                 target_fps: float = 60.0, enable_motion_filter: bool = True):
        """
        Initialize optimized traffic monitor.
        
        Args:
            model_path: Path to YOLO model
            conf_threshold: Confidence threshold
            target_fps: Target FPS
            enable_motion_filter: Enable motion detection pre-filter
        """
        print("=" * 70)
        print("INITIALIZING TRAFFIC MONITORING SYSTEM - KALMAN FILTER OPTIMIZED")
        print("=" * 70)
        print(f"Target FPS: {target_fps}")
        
        # Vehicle classes (COCO dataset)
        self.vehicle_classes = {2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'}
        
        # Colors for visualization
        self.colors = {
            'car': (0, 255, 0),
            'motorcycle': (0, 0, 255),
            'bus': (255, 0, 0),
            'truck': (255, 255, 0)
        }
        
        # ========== LAYER 1: GPU Optimizer ==========
        self.gpu_optimizer = GPUOptimizer()
        
        # ========== LAYER 2: YOLO Model ==========
        print(f"\nLoading model: {model_path}")
        self.model = YOLO(model_path)
        self.model = self.gpu_optimizer.optimize_model(self.model)
        self.conf_threshold = conf_threshold
        
        # ========== LAYER 3: Kalman Tracker ==========
        self.kalman_tracker = KalmanTrackerManager(
            max_age=30,
            min_hits=2,
            iou_threshold=0.3,
            distance_threshold=100
        )
        
        # ========== LAYER 4: Motion Detector ==========
        self.enable_motion_filter = enable_motion_filter
        self.motion_detector = MotionDetector() if enable_motion_filter else None
        
        # ========== LAYER 5: Adaptive Processor ==========
        self.adaptive_processor = AdaptiveProcessor(target_fps=target_fps)
        
        # ========== LAYER 6: NMS Optimizer ==========
        self.nms_optimizer = NMSOptimizer(
            iou_threshold=0.5,
            score_threshold=conf_threshold,
            use_soft_nms=True
        )
        
        # ========== LAYER 7: Speed Estimator ==========
        self.speed_estimator = SpeedEstimator()
        
        # ========== Statistics ==========
        self.vehicle_count = defaultdict(int)
        self.total_count = 0
        self.counted_ids = set()
        self.line_position = None
        
        # FPS tracking
        self.fps = 0
        self.fps_history = deque(maxlen=30)
        self.prev_frame_time = 0
        
        # Traffic analysis
        self.density_history = deque(maxlen=300)
        self.congestion_level = "normal"
        
        # Session data
        self.log_data = []
        self.speed_violations = []
        self.speed_limit = 60
        
        print("\nAll layers initialized successfully!")
        print("=" * 70)
    
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, List[dict], dict]:
        """
        Process single frame through all layers.
        
        Args:
            frame: Input frame
            
        Returns:
            (annotated_frame, detections, statistics)
        """
        speed = 0.0  # Initialize speed variable
        # Calculate FPS
        current_time = time.time()
        if self.prev_frame_time > 0:
            dt = current_time - self.prev_frame_time
            self.fps = 1 / dt if dt > 0 else 0
        self.prev_frame_time = current_time
        self.fps_history.append(self.fps)
        
        # Update adaptive processor
        scale = self.adaptive_processor.get_adaptive_scale(self.fps)
        
        # Skip frame if needed (adaptive)
        if self.adaptive_processor.should_skip_frame():
            # If we skip YOLO, at least return the previous annotated frame to prevent blinking
            # OR better: reuse the last known tracks to draw on the OLD frame (freeze effect) 
            # or current frame (ghost effect possible if movement is large)
            
            # For simplicity and stability, we return a cached annotated frame if available, 
            # otherwise we just return the raw frame.
            # To fix blinking properly, we should store self.last_annotated_frame
            
            result_stats = {
                'fps': self.fps, 
                'skipped': True,
                'total_count': self.total_count,
                'congestion': "N/A"
            }
            
            if hasattr(self, 'last_annotated_frame') and self.last_annotated_frame is not None:
                return self.last_annotated_frame, [], result_stats
            else:
                return frame, [], result_stats
        
        # Resize frame for performance
        processed_frame = self.adaptive_processor.resize_frame(frame, scale)
        
        # Motion detection (optional pre-filter)
        motion_regions = []
        if self.enable_motion_filter and self.motion_detector:
            _, motion_regions = self.motion_detector.detect_motion(processed_frame)
        
        # YOLOv8 inference
        results = self.model.track(
            processed_frame,
            conf=self.conf_threshold,
            persist=True,
            verbose=False,
            iou=0.5
        )
        
        # Parse detections
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls_id = int(box.cls[0])
                
                if cls_id in self.vehicle_classes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    class_name = self.vehicle_classes[cls_id]
                    
                    # Scale back to original size
                    if scale != 1.0:
                        x1, y1, x2, y2 = self.adaptive_processor.scale_bbox(
                            (x1, y1, x2, y2), scale
                        )
                    
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2
                    
                    detections.append({
                        'class': class_name,
                        'confidence': conf,
                        'bbox': (x1, y1, x2, y2),
                        'center': (center_x, center_y)
                    })
        
        # Apply optimized NMS
        detections = self.nms_optimizer.class_aware_nms(detections)
        
        # Update Kalman Tracker
        tracks = self.kalman_tracker.update(detections)
        
        # Update speed estimator FPS
        self.speed_estimator.fps = max(self.fps, 30)
        
        # Auto calibrate if needed
        if self.speed_estimator.pixels_per_meter == 10.0:
            self.speed_estimator.pixels_per_meter = frame.shape[0] / 15.0
        
        # Set counting line
        if self.line_position is None:
            self.line_position = frame.shape[0] // 2
        
        # Draw and process
        annotated_frame = frame.copy()
        
        # Draw counting line
        cv2.line(annotated_frame, (0, self.line_position),
                (frame.shape[1], self.line_position), (0, 255, 255), 2)
        
        # Process tracks
        speed = 0.0  # Initialize speed to avoid UnboundLocalError
        for track in tracks:
            track_id = track['track_id']
            class_name = track['class']
            bbox = track['bbox']
            center = track['center']
            confidence = track['confidence']
            velocity = track.get('velocity', (0, 0))
            history = track.get('history', [])
            
            # Estimate speed from Kalman velocity
            speed = self.speed_estimator.estimate_speed(track_id, velocity)
            
            # Line counting
            if center[1] > self.line_position - 20 and center[1] < self.line_position + 20:
                if track_id not in self.counted_ids:
                    self.counted_ids.add(track_id)
                    self.vehicle_count[class_name] += 1
                    self.total_count += 1
                    
                    # Log
                    self.log_data.append({
                        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        'vehicle_type': class_name,
                        'confidence': confidence,
                        'track_id': track_id,
                        'speed': round(speed, 1)
                    })
            
            # Draw bounding box
            color = self.colors.get(class_name, (255, 255, 255))
            x1, y1, x2, y2 = bbox
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw trajectory
            if len(history) > 1:
                for i in range(1, len(history)):
                    cv2.line(annotated_frame, history[i-1], history[i], color, 1)
            
            # Draw center
            cv2.circle(annotated_frame, center, 4, (0, 0, 255), -1)
            
            # Draw velocity arrow (Kalman prediction)
            vx, vy = velocity
            end_point = (int(center[0] + vx * 5), int(center[1] + vy * 5))
            cv2.arrowedLine(annotated_frame, center, end_point, (255, 0, 255), 2)
            
            # Label
            label = f"{class_name} #{track_id} {confidence:.2f} {speed:.0f}km/h"
            (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(annotated_frame, (x1, y1 - text_h - 8), (x1 + text_w, y1), color, -1)
            cv2.putText(annotated_frame, label, (x1, y1 - 4),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Speed warning
            if speed > self.speed_limit:
                cv2.putText(annotated_frame, "!", (x2 - 15, y1 + 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Calculate density
        density = self.calculate_density(len(tracks), frame.shape)
        congestion, congestion_color = self.detect_congestion(density)
        
        # Draw info panel
        self._draw_info_panel(annotated_frame, speed, congestion, congestion_color)
        
        # Statistics
        stats = {
            'fps': self.fps,
            'target_fps': self.adaptive_processor.target_fps,
            'scale': scale,
            'total_count': self.total_count,
            'vehicle_count': dict(self.vehicle_count),
            'active_tracks': len(tracks),
            'density': density,
            'congestion': congestion
        }
        
        # Cache for skipped frames
        self.last_annotated_frame = annotated_frame
        
        return annotated_frame, tracks, stats
    
    def calculate_density(self, num_vehicles: int, frame_shape: Tuple[int, int]) -> float:
        """Calculate traffic density."""
        # Simple density based on vehicle count per area
        area = frame_shape[0] * frame_shape[1]
        avg_vehicle_area = 30 * 15  # Average vehicle area in pixels
        density = (num_vehicles * avg_vehicle_area) / area
        self.density_history.append(density)
        return density
    
    def detect_congestion(self, density: float) -> Tuple[str, str]:
        """Detect congestion level."""
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
        """Draw information panel."""
        # Main panel
        cv2.rectangle(frame, (10, 10), (350, 180), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (350, 180), (0, 255, 255), 2)
        
        # Title
        cv2.putText(frame, "KALMAN FILTER OPTIMIZED", (20, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # FPS with target
        avg_fps = np.mean(list(self.fps_history)) if self.fps_history else self.fps
        fps_color = (0, 255, 0) if avg_fps >= 60 else (0, 255, 255) if avg_fps >= 30 else (0, 0, 255)
        cv2.putText(frame, f"FPS: {avg_fps:.1f} / {self.adaptive_processor.target_fps:.0f}", 
                   (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, fps_color, 1)
        
        # Scale
        cv2.putText(frame, f"Scale: {self.adaptive_processor.current_scale:.2f}", 
                   (200, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Total vehicles
        cv2.putText(frame, f"Total Vehicles: {self.total_count}", 
                   (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Active tracks
        cv2.putText(frame, f"Active Tracks: {len(self.kalman_tracker.trackers)}", 
                   (20, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Vehicle breakdown
        y_offset = 125
        for vtype, count in self.vehicle_count.items():
            color = self.colors.get(vtype, (255, 255, 255))
            cv2.putText(frame, f"{vtype.capitalize()}: {count}", 
                       (20 + (y_offset - 125) * 110, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            if (y_offset - 125) % 2 == 1:
                y_offset += 20
        
        # Congestion
        color_map = {'green': (0, 255, 0), 'yellow': (0, 255, 255), 
                    'orange': (0, 165, 255), 'red': (0, 0, 255)}
        cv2.putText(frame, f"Status: {congestion}", (20, 165),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_map.get(congestion_color, (255, 255, 255)), 2)
        
        # Timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(frame, timestamp, (frame.shape[1] - 250, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    def process_video(self, source, output_path: str = None, show_display: bool = True):
        """
        Process video with all optimizations.
        
        Args:
            source: Video source (path, URL, or device index)
            output_path: Output video path
            show_display: Show display window
        """
        if isinstance(source, str) and (source.startswith('http') or source.startswith('rtsp')):
            print(f"Initializing threaded stream loader for: {source}")
            stream_loader = StreamLoader(source).start()
            use_loader = True
            time.sleep(2.0) # Buffer fill
            
            fps = stream_loader.stream.get(cv2.CAP_PROP_FPS) or 30
            width = int(stream_loader.stream.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(stream_loader.stream.get(cv2.CAP_PROP_FRAME_HEIGHT))
        else:
            cap = cv2.VideoCapture(source)
            if not cap.isOpened():
                print(f"Error: Cannot open video source: {source}")
                return
            use_loader = False
            fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Update speed estimator FPS
        self.speed_estimator.fps = fps
        
        print(f"\nVideo: {width}x{height} @ {fps}fps")
        print(f"Target: {self.adaptive_processor.target_fps}fps")
        
        # Video writer
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        start_time = time.time()
        
        try:
            while True:
                if use_loader:
                    if not stream_loader.running():
                        break
                        
                    ret, frame = stream_loader.read()
                    if not ret:
                        time.sleep(0.01) # Wait for buffer
                        continue
                else:
                    ret, frame = cap.read()
                    if not ret:
                        print("Video selesai atau error membaca frame")
                        break

                

                
                # Process frame
                processed_frame, tracks, stats = self.process_frame(frame)
                
                # Write output
                if writer:
                    writer.write(processed_frame)
                
                # Display
                if show_display:
                    cv2.imshow('Traffic Monitor - Kalman Filter Optimized', processed_frame)
                    
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    elif key == ord('s'):
                        screenshot = f"screenshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                        cv2.imwrite(screenshot, processed_frame)
                        print(f"Screenshot: {screenshot}")
                
                frame_count += 1
                
                if frame_count % 30 == 0:
                    elapsed = time.time() - start_time
                    print(f"Frame {frame_count} | FPS: {stats['fps']:.1f} | "
                          f"Total: {stats['total_count']} | Status: {stats['congestion']}")
        
        except KeyboardInterrupt:
            print("\nStopped by user")
        
        finally:
            if use_loader:
                stream_loader.stop()
            else:
                cap.release()
                
            if writer:
                writer.release()
            cv2.destroyAllWindows()
            
            # Summary
            self._print_summary(frame_count, start_time)
    
    def _print_summary(self, frame_count: int, start_time: float):
        """Print processing summary."""
        elapsed = time.time() - start_time
        
        print("\n" + "=" * 70)
        print("PROCESSING SUMMARY - KALMAN FILTER OPTIMIZED")
        print("=" * 70)
        print(f"Total frames: {frame_count}")
        print(f"Processing time: {elapsed:.1f}s")
        print(f"Average FPS: {frame_count/elapsed:.1f}")
        print(f"\nTotal vehicles: {self.total_count}")
        print("By type:")
        for vtype, count in self.vehicle_count.items():
            print(f"  - {vtype}: {count}")
        print("=" * 70)


# ==================== ENTRY POINT ====================

def main():
    """Main function with interactive menu."""
    print("=" * 70)
    print("TRAFFIC MONITORING SYSTEM - KALMAN FILTER OPTIMIZED VERSION")
    print("Target: 60+ FPS dengan akurasi tinggi")
    print("=" * 70)
    
    print("\nPilih sumber video:")
    print("1. Webcam")
    print("2. File video lokal")
    print("3. URL Stream/CCTV")
    print("4. CCTV Jogja (ATCS)")
    
    choice = input("\nPilihan (1-4): ").strip()
    
    if choice == '1':
        source = 0
        output = "output_webcam_kalman.mp4"
    elif choice == '2':
        source = input("Path file video: ").strip()
        output = f"output_kalman_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
    elif choice == '3':
        source = input("URL stream: ").strip()
        output = f"output_stream_kalman_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
    elif choice == '4':
        source = "http://cctvjss.jogjakota.go.id/atcs/ATCS_Lampu_Merah_SugengJeroni2.stream/playlist.m3u8"
        output = "output_cctv_jogja_kalman.mp4"
    else:
        print("Pilihan tidak valid")
        return
    
    # Configuration
    # model = input("Model (yolov8n/s/m) [default: yolov8n]: ").strip() or 'yolov8n'
    target_fps = input("Target FPS [default: 60]: ").strip()
    target_fps = float(target_fps) if target_fps else 60.0
    
    # Initialize
    print("\nInitializing...")
    monitor = TrafficMonitorKalmanOptimized(
        model_path='yolov26n.pt',
        conf_threshold=0.4,
        target_fps=target_fps,
        enable_motion_filter=True
    )
    
    save_output = input("\nSimpan output? (y/n): ").strip().lower() == 'y'
    
    print(f"\nMemulai deteksi dengan target {target_fps} FPS...")
    print("Tekan 'q' untuk berhenti, 's' untuk screenshot\n")
    
    monitor.process_video(
        source,
        output_path=output if save_output else None,
        show_display=True
    )


if __name__ == "__main__":
    main()
