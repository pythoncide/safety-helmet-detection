"""
Real-time YouTube Object Detection with Faster R-CNN
Universal class support for any trained model
Synchronized to original video FPS for real-time streaming simulation
"""

import torch
import cv2
import numpy as np
from PIL import Image
import argparse
import time
import hashlib
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision import transforms
import yt_dlp

# Import preprocessing functions
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils.dataset2 import letterbox_resize, inverse_transform_bbox

# ============================================================================
# CONFIGURATION - 학습된 모델에 맞게 수정하세요!
# ============================================================================

CONFIG = {
    # Model configuration
    #'model_path': 'Entire_Model.pth',  # Model file path
    'model_path': 'safety_helmet_model2.pth',  # Model file path
    'class_names': ['O', 'X'],  # Classes (WITHOUT '__background__')
    
    # Detection parameters
    #'conf_threshold': 0.5,   # Initial confidence threshold
    #'iou_threshold': 0.5,    # Initial IOU threshold for NMS
    #'resize_size': 640,      # Input size for model
    'resize_size': 640,      # Input size for model
    'conf_threshold': 0.95   ,   # Initial confidence threshold
    'iou_threshold': 0.45,    # Initial IOU threshold for NMS
    
    # Video settings
    'cache_dir': 'video_cache',  # Cache directory for downloaded videos
}

# ============================================================================
# EXAMPLE CONFIGURATIONS FOR DIFFERENT MODELS
# ============================================================================

# Example 1: COCO Person & Car (default)
# CONFIG = {
#     'model_path': 'fasterrcnn_model.pth',
#     'class_names': ['person', 'car'],
#     'conf_threshold': 0.5,
#     ...
# }

# Example 2: Animal Detection
# CONFIG = {
#     'model_path': 'animal_detector.pth',
#     'class_names': ['cat', 'dog', 'bird'],
#     'conf_threshold': 0.6,
#     ...
# }

# Example 3: Traffic Detection
# CONFIG = {
#     'model_path': 'traffic_detector.pth',
#     'class_names': ['car', 'truck', 'bus', 'motorcycle', 'bicycle'],
#     'conf_threshold': 0.5,
#     ...
# }

# Example 4: Defect Detection
# CONFIG = {
#     'model_path': 'defect_detector.pth',
#     'class_names': ['crack', 'rust', 'dent', 'scratch'],
#     'conf_threshold': 0.7,
#     ...
# }

# ============================================================================

device = torch.device('cuda' if torch.cuda.is_available() else 'xpu')
print(device)

def generate_colors(num_classes):
    """
    Generate distinct colors for each class
    Returns: dict mapping class_id (1-indexed) to BGR color
    """
    np.random.seed(42)  # Fixed seed for consistent colors
    colors = {}
    
    # Predefined colors for common classes
    predefined_colors = [
        (255, 0, 0),      # Blue
        (0, 0, 255),      # Red
        (0, 255, 0),      # Green
        (255, 255, 0),    # Cyan
        (255, 0, 255),    # Magenta
        (0, 255, 255),    # Yellow
        (128, 0, 128),    # Purple
        (255, 165, 0),    # Orange
        (0, 128, 128),    # Teal
        (128, 128, 0),    # Olive
    ]
    
    for i in range(1, num_classes + 1):
        if i - 1 < len(predefined_colors):
            colors[i] = predefined_colors[i - 1]
        else:
            # Generate random distinct color
            colors[i] = tuple(np.random.randint(50, 255, 3).tolist())
    
    return colors


def validate_model_classes(model, expected_num_classes):
    """
    Validate that model's number of classes matches expected
    
    Args:
        model: Faster R-CNN model
        expected_num_classes: Expected number of classes (including background)
    
    Returns:
        bool: True if valid, False otherwise
    """
    try:
        # Get model's number of output classes
        model_num_classes = model.roi_heads.box_predictor.cls_score.out_features
        
        if model_num_classes != expected_num_classes:
            print(f"\n⚠️  WARNING: Model class mismatch!")
            print(f"  Expected classes: {expected_num_classes} (including background)")
            print(f"  Model classes: {model_num_classes}")
            print(f"  Please check your CONFIG['class_names']")
            return False
        
        return True
    except Exception as e:
        print(f"\n⚠️  Warning: Could not validate model classes: {e}")
        return True  # Continue anyway


def get_model_dtype(model):
    """Detect model's dtype (FP32 or FP16)"""
    return next(model.parameters()).dtype


def download_youtube_video(youtube_url, cache_dir, force_download=False):
    """
    Download YouTube video with caching support
    Returns: path to video file (cached or newly downloaded)
    """
    print(f"\nExtracting video ID...")
    try:
        with yt_dlp.YoutubeDL({'quiet': True}) as ydl:
            info = ydl.extract_info(youtube_url, download=False)
            video_id = info.get('id', None)
    except:
        video_id = hashlib.md5(youtube_url.encode()).hexdigest()[:11]
    
    print(f"Video ID: {video_id}")
    
    # Check cache
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    cached_path = os.path.join(cache_dir, f"{video_id}.mp4")
    
    if os.path.exists(cached_path) and not force_download:
        file_size = os.path.getsize(cached_path) / (1024 * 1024)
        print(f"\n✓ Found cached video!")
        print(f"  Path: {cached_path}")
        print(f"  Size: {file_size:.1f} MB")
        print(f"  Skipping download...")
        return cached_path
    
    # Download video
    print(f"\nDownloading YouTube video...")
    print(f"This will take 1-2 minutes depending on video length and your internet speed")
    
    ydl_opts = {
        'format': (
            'best[height<=720][ext=mp4][vcodec^=avc1]/best[height<=720][ext=mp4]/'
            'bestvideo[height<=720][ext=mp4]/bestvideo[height<=720]/'
            'best[height<=720]'
        ),
        'outtmpl': cached_path,
        'quiet': False,
        'no_warnings': True,
        'noplaylist': True,
        'prefer_free_formats': False,
        'progress_hooks': [lambda d: print(f"\rDownload: {d.get('_percent_str', 'N/A')} ", end='') 
                          if d['status'] == 'downloading' else None],
    }
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            print(f"\nFetching video info...")
            info = ydl.extract_info(youtube_url, download=True)
            video_title = info.get('title', 'Unknown')
            duration = info.get('duration', 0)
            
        if not os.path.exists(cached_path):
            raise FileNotFoundError(f"Downloaded file not found: {cached_path}")
        
        file_size = os.path.getsize(cached_path) / (1024 * 1024)
        
        print(f"\n✓ Video downloaded successfully")
        print(f"  Title: {video_title}")
        print(f"  Duration: {duration // 60}m {duration % 60}s")
        print(f"  File size: {file_size:.1f} MB")
        print(f"  Cached to: {cached_path}")
        return cached_path
        
    except Exception as e:
        print(f"\nError: Failed to download video")
        print(f"Error details: {e}")
        raise


def preprocess_frame(frame, resize_size=640, model_dtype=torch.float32):
    """
    Preprocess frame for model inference with automatic dtype conversion
    
    Args:
        frame: OpenCV BGR frame
        resize_size: target size for letterbox resize
        model_dtype: dtype of the model (torch.float32 or torch.float16)
    
    Returns: 
        tensor, scale, pad_left, pad_top
    """
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(frame_rgb)
    
    img_letterbox, scale, pad_left, pad_top = letterbox_resize(
        pil_image, target_size=resize_size
    )
    
    normalize = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    img_tensor = normalize(img_letterbox).unsqueeze(0).to(device)
    
    # Convert to FP16 if model is FP16
    if model_dtype == torch.float16:
        img_tensor = img_tensor.half()
    
    return img_tensor, scale, pad_left, pad_top


def postprocess_predictions(predictions, scale, pad_left, pad_top, 
                           conf_threshold=0.5, frame_shape=None):
    """
    Postprocess model predictions
    Returns: filtered boxes, labels, scores in original image coordinates
    """
    pred = predictions[0]
    boxes = pred['boxes'].cpu().numpy()
    labels = pred['labels'].cpu().numpy()
    scores = pred['scores'].cpu().numpy()
    
    # Filter by confidence
    mask = scores >= conf_threshold
    boxes = boxes[mask]
    labels = labels[mask]
    scores = scores[mask]
    
    # Transform boxes back to original coordinates
    transformed_boxes = []
    for box in boxes:
        orig_box = inverse_transform_bbox(box, scale, pad_left, pad_top)
        transformed_boxes.append(orig_box)
    
    # Clip boxes to frame boundaries if frame_shape provided
    if frame_shape is not None and len(transformed_boxes) > 0:
        h, w = frame_shape[:2]
        transformed_boxes = np.array(transformed_boxes)
        transformed_boxes[:, [0, 2]] = np.clip(transformed_boxes[:, [0, 2]], 0, w)
        transformed_boxes[:, [1, 3]] = np.clip(transformed_boxes[:, [1, 3]], 0, h)
    
    return transformed_boxes, labels, scores


def draw_predictions(frame, boxes, labels, scores, class_names, class_colors, conf_threshold):
    """Draw bounding boxes and labels on frame"""
    for box, label, score in zip(boxes, labels, scores):
        x1, y1, x2, y2 = map(int, box)
        color = class_colors.get(label, (0, 255, 0))
        class_name = class_names[label]
        
        # Draw thicker, brighter bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
        
        # Add semi-transparent fill for better visibility
        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
        cv2.addWeighted(overlay, 0.1, frame, 0.9, 0, frame)
        
        label_text = f'{class_name}: {score:.2f}'
        (text_w, text_h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        
        # Draw label background with more visibility
        cv2.rectangle(frame, (x1, y1 - text_h - 12), (x1 + text_w + 4, y1), color, -1)
        
        # Draw label text with black outline for better readability
        cv2.putText(frame, label_text, (x1 + 2, y1 - 6),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
        cv2.putText(frame, label_text, (x1 + 2, y1 - 6),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    return frame


def draw_info_panel(frame, class_counts, class_names, inference_fps, conf_threshold, 
                   iou_threshold, video_fps, processing_rate, dropped_frames, 
                   realtime_capable, model_dtype):
    """Draw information panel on frame with dynamic class display"""
    overlay = frame.copy()
    h, w = frame.shape[:2]
    
    # Calculate panel height based on number of classes
    num_classes = len(class_names) - 1  # Exclude background
    base_height = 280
    class_height = num_classes * 22
    panel_height = min(base_height + class_height, h - 100)
    
    # Main info panel
    cv2.rectangle(overlay, (10, 10), (380, panel_height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    
    # Real-time status indicator
    status_color = (0, 255, 0) if realtime_capable else (0, 0, 255)
    status_text = "REAL-TIME ✓" if realtime_capable else "DROPPING FRAMES ✗"
    
    # Model dtype display
    dtype_str = "FP16" if model_dtype == torch.float16 else "FP32"
    dtype_color = (255, 165, 0) if model_dtype == torch.float16 else (255, 255, 255)
    
    # Build info texts with dynamic class counts
    info_texts = []
    
    # Display each class count
    total_count = 0
    for class_id, count in sorted(class_counts.items()):
        class_name = class_names[class_id]
        info_texts.append(f"{class_name}: {count}")
        total_count += count
    
    info_texts.append(f"Total: {total_count}")
    info_texts.append("")
    info_texts.append(f"Model: {dtype_str}")
    info_texts.append(f"Video FPS: {video_fps:.1f}")
    info_texts.append(f"Inference FPS: {inference_fps:.1f}")
    info_texts.append(f"Processing Rate: {processing_rate:.1f}%")
    info_texts.append(f"Dropped Frames: {dropped_frames}")
    info_texts.append("")
    info_texts.append(f"Confidence: {conf_threshold:.2f}")
    info_texts.append(f"IOU: {iou_threshold:.2f}")
    
    y_offset = 35
    for i, text in enumerate(info_texts):
        if text:
            # Color coding
            if "Model:" in text:
                color = dtype_color
            elif "Video FPS:" in text:
                color = (100, 255, 255)
            elif "Inference FPS:" in text:
                color = (255, 255, 100)
            elif "Processing Rate:" in text:
                color = (0, 255, 0) if processing_rate >= 99 else (0, 165, 255)
            else:
                color = (255, 255, 255)
            
            cv2.putText(frame, text, (20, y_offset + i * 22),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    
    # Real-time status
    cv2.putText(frame, status_text, (20, y_offset + len(info_texts) * 22),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
    
    # Controls info at bottom
    controls = [
        "Controls: [+/-] Conf | [[/]] IOU | [SPACE] Pause | [ESC] Exit"
    ]
    
    for i, text in enumerate(controls):
        cv2.putText(frame, text, (10, h - 20 - i * 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return frame


class RealtimePerformanceTracker:
    """Track real-time processing performance"""
    def __init__(self, video_fps):
        self.video_fps = video_fps
        self.frame_interval = 1.0 / video_fps if video_fps > 0 else 0.033
        self.total_frames = 0
        self.processed_frames = 0
        self.dropped_frames = 0
        self.inference_times = []
        self.max_history = 30
        
    def update(self, processing_time, frames_to_skip=0):
        """Update performance metrics"""
        self.total_frames += 1
        self.processed_frames += 1
        self.dropped_frames += frames_to_skip
        
        self.inference_times.append(processing_time)
        if len(self.inference_times) > self.max_history:
            self.inference_times.pop(0)
    
    def get_inference_fps(self):
        """Calculate average inference FPS"""
        if not self.inference_times:
            return 0.0
        avg_time = np.mean(self.inference_times)
        return 1.0 / avg_time if avg_time > 0 else 0.0
    
    def get_processing_rate(self):
        """Calculate real-time processing rate"""
        if self.total_frames == 0:
            return 100.0
        return (self.processed_frames / (self.processed_frames + self.dropped_frames)) * 100
    
    def is_realtime_capable(self):
        """Check if currently processing in real-time"""
        if not self.inference_times:
            return True
        avg_time = np.mean(self.inference_times[-10:])
        return avg_time < self.frame_interval


def main():
    parser = argparse.ArgumentParser(description='Real-time YouTube Object Detection')
    parser.add_argument('--url', type=str, required=True, help='YouTube URL')
    parser.add_argument('--force-download', action='store_true',
                       help='Force re-download even if cached')
    args = parser.parse_args()
    
    print("="*70)
    print("Universal Real-time YouTube Object Detection")
    print("="*70)
    print(f"Device: {device}")
    
    # Print configuration
    print("\nConfiguration:")
    print(f"  Model: {CONFIG['model_path']}")
    print(f"  Classes: {CONFIG['class_names']}")
    print(f"  Confidence threshold: {CONFIG['conf_threshold']}")
    print(f"  IOU threshold: {CONFIG['iou_threshold']}")
    print(f"  Resize size: {CONFIG['resize_size']}")
    print(f"  Cache directory: {CONFIG['cache_dir']}")
    
    # Prepare class names with background
    class_names = ['__background__'] + CONFIG['class_names']
    num_classes = len(class_names)
    
    print(f"  Total classes (with background): {num_classes}")
    
    # Generate colors for classes
    class_colors = generate_colors(len(CONFIG['class_names']))
    print(f"  Generated {len(class_colors)} class colors")
    
    # Load model
    print(f"\nLoading model...")
    if not os.path.exists(CONFIG['model_path']):
        raise FileNotFoundError(f"Model not found: {CONFIG['model_path']}")
    
    model = torch.load(CONFIG['model_path'], map_location=device, weights_only=False)
    model.to(device)
    model.eval()
    
    # Detect model dtype
    model_dtype = get_model_dtype(model)
    dtype_str = "FP16" if model_dtype == torch.float16 else "FP32"
    print(f"✓ Model loaded")
    print(f"  Model dtype: {dtype_str}")
    
    # Validate model classes
    if not validate_model_classes(model, num_classes):
        print("\n⚠️  Continuing anyway, but results may be incorrect!")
        print("  Make sure CONFIG['class_names'] matches your trained model")
    
    # Download video (with caching)
    video_path = download_youtube_video(args.url, CONFIG['cache_dir'], 
                                       force_download=args.force_download)
    
    # Open video file
    print("\nOpening video file...")
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Error: Could not open video file")
        return
    
    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print("✓ Video file opened successfully")
    print(f"  Resolution: {width}x{height}")
    print(f"  Video FPS: {video_fps:.1f} ← Playback synchronized to this")
    print(f"  Total frames: {total_frames}")
    if video_fps > 0 and total_frames > 0:
        duration = total_frames / video_fps
        print(f"  Duration: {int(duration // 60)}m {int(duration % 60)}s")
        print(f"  Frame interval: {1000/video_fps:.1f}ms per frame")
    
    # Test reading first frame
    print("\nTesting frame reading...")
    ret, test_frame = cap.read()
    if not ret or test_frame is None:
        print("✗ ERROR: Cannot read video frames!")
        cap.release()
        return
    else:
        print(f"✓ Successfully read first frame: {test_frame.shape}")
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    print("\nStarting real-time detection...")
    print("Press [+/-] to adjust confidence threshold")
    print("Press [[/]] to adjust IOU threshold")
    print("Press [SPACE] to pause/resume")
    print("Press [ESC] to exit")
    print("-"*70)
    
    # Initialize variables
    conf_threshold = CONFIG['conf_threshold']
    iou_threshold = CONFIG['iou_threshold']
    performance_tracker = RealtimePerformanceTracker(video_fps)
    frame_count = 0
    paused = False
    last_frame = None
    
    frame_interval = 1.0 / video_fps if video_fps > 0 else 0.033
    next_frame_time = time.time()
    
    try:
        while True:
            current_time = time.time()
            
            if not paused:
                # Check if it's time for the next frame
                if current_time >= next_frame_time:
                    ret, frame = cap.read()
                    if not ret:
                        if frame_count == 0:
                            print("\n✗ Error: Could not read first frame!")
                        else:
                            print(f"\n✓ Video completed")
                            print(f"  Total frames: {frame_count}")
                            print(f"  Processed: {performance_tracker.processed_frames}")
                            print(f"  Dropped: {performance_tracker.dropped_frames}")
                            print(f"  Processing rate: {performance_tracker.get_processing_rate():.1f}%")
                        break
                    
                    if frame is None:
                        frame_count += 1
                        next_frame_time += frame_interval
                        continue
                    
                    frame_count += 1
                    process_start = time.time()
                    
                    # Update model's NMS threshold
                    model.roi_heads.nms_thresh = iou_threshold
                    
                    # Preprocess with automatic dtype conversion
                    img_tensor, scale, pad_left, pad_top = preprocess_frame(
                        frame, resize_size=CONFIG['resize_size'], model_dtype=model_dtype
                    )
                    
                    # Inference
                    with torch.no_grad():
                        predictions = model(img_tensor)
                    
                    # Postprocess
                    boxes, labels, scores = postprocess_predictions(
                        predictions, scale, pad_left, pad_top,
                        conf_threshold=conf_threshold,
                        frame_shape=frame.shape
                    )
                    
                    # Debug output - show what's detected
                    if frame_count % 30 == 0:  # Print every 30 frames
                        print(f"\n[Frame {frame_count}] Detected: {len(boxes)} objects")
                        if len(boxes) > 0:
                            for i, (label, score) in enumerate(zip(labels, scores)):
                                class_name = class_names[label]
                                print(f"  {i+1}. {class_name}: {score:.3f}")
                        else:
                            print(f"  No objects detected (conf threshold: {conf_threshold:.2f})")
                    
                    # Count objects by class
                    class_counts = {}
                    for class_id in range(1, num_classes):  # Skip background
                        count = np.sum(labels == class_id)
                        if count > 0:
                            class_counts[class_id] = count
                    
                    # Draw predictions
                    frame = draw_predictions(frame, boxes, labels, scores, 
                                           class_names, class_colors, conf_threshold)
                    
                    # --- [추가 코드 시작] ------------------------------------
                    # 'X' 클래스(=안전모 미착용) 감지 시 경고 메시지 표시
                    if any(class_names[label] == 'X' for label in labels):
                        cv2.putText(frame, "안전모 미착용 감지", (50, 80),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 4)
                    # --- [추가 코드 끝] -------------------------------------

                    
                    processing_time = time.time() - process_start
                    
                    # Calculate frames to skip if processing is too slow
                    frames_to_skip = 0
                    if processing_time > frame_interval:
                        frames_behind = int(processing_time / frame_interval)
                        frames_to_skip = frames_behind
                        # Skip frames to catch up
                        for _ in range(frames_to_skip):
                            cap.read()
                            frame_count += 1
                    
                    # Update performance tracker
                    performance_tracker.update(processing_time, frames_to_skip)
                    
                    # Draw info panel
                    frame = draw_info_panel(
                        frame, class_counts, class_names,
                        performance_tracker.get_inference_fps(),
                        conf_threshold, iou_threshold, video_fps,
                        performance_tracker.get_processing_rate(),
                        performance_tracker.dropped_frames,
                        performance_tracker.is_realtime_capable(),
                        model_dtype
                    )
                    
                    last_frame = frame
                    
                    # Schedule next frame
                    next_frame_time += frame_interval
                    
                    # Adjust if we're too far behind
                    if next_frame_time < current_time:
                        next_frame_time = current_time
            
            # Display frame
            if last_frame is not None:
                cv2.imshow('YouTube Real-time Detection', last_frame)
            
            # Calculate wait time
            time_until_next_frame = next_frame_time - time.time()
            wait_ms = max(1, int(time_until_next_frame * 1000))
            
            # Handle keyboard input
            key = cv2.waitKey(wait_ms) & 0xFF
            
            if key == 27:  # ESC
                print("\nExiting...")
                break
            elif key == ord(' '):  # SPACE
                paused = not paused
                status = "PAUSED" if paused else "RESUMED"
                print(f"\n{status}")
                if not paused:
                    next_frame_time = time.time()
            elif key == ord('+') or key == ord('='):
                conf_threshold = min(conf_threshold + 0.05, 0.95)
                print(f"\nConfidence threshold: {conf_threshold:.2f}")
            elif key == ord('-') or key == ord('_'):
                conf_threshold = max(conf_threshold - 0.05, 0.05)
                print(f"\nConfidence threshold: {conf_threshold:.2f}")
            elif key == ord(']'):
                iou_threshold = min(iou_threshold + 0.05, 0.95)
                print(f"\nIOU threshold: {iou_threshold:.2f}")
            elif key == ord('['):
                iou_threshold = max(iou_threshold - 0.05, 0.30)
                print(f"\nIOU threshold: {iou_threshold:.2f}")
    
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        
        print("\n" + "="*70)
        print("Performance Summary")
        print("="*70)
        print(f"Model: {dtype_str}")
        print(f"Classes: {CONFIG['class_names']}")
        print(f"Video FPS: {video_fps:.1f}")
        print(f"Average Inference FPS: {performance_tracker.get_inference_fps():.1f}")
        print(f"Real-time Processing Rate: {performance_tracker.get_processing_rate():.1f}%")
        print(f"Total Frames: {frame_count}")
        print(f"Processed Frames: {performance_tracker.processed_frames}")
        print(f"Dropped Frames: {performance_tracker.dropped_frames}")
        if performance_tracker.is_realtime_capable():
            print("Status: ✓ REAL-TIME CAPABLE")
        else:
            print("Status: ✗ NOT REAL-TIME (Consider model optimization)")
        print(f"\nVideo cached at: {video_path}")
        print("="*70)


if __name__ == "__main__":
    main()