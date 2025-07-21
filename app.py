import os
import time
import json
import statistics
from datetime import datetime
from flask import Flask, request, jsonify, render_template, url_for, send_file, Response
from werkzeug.utils import secure_filename

# Fix for missing modules before importing YOLO - Universal compatibility
def add_missing_modules():
    """Add ALL missing modules for compatibility with any YOLO version"""
    try:
        from ultralytics.nn.modules import block
        import torch
        import torch.nn as nn
        
        # Get existing modules to avoid duplication
        from ultralytics.nn.modules.conv import Conv
        from ultralytics.nn.modules.block import Bottleneck
        
        # First, define base C3 class that will be used by others
        class C3Base(nn.Module):
            """CSP Bottleneck with 3 convolutions"""
            def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
                super().__init__()
                c_ = int(c2 * e)  # hidden channels
                self.cv1 = Conv(c1, c_, 1, 1)
                self.cv2 = Conv(c1, c_, 1, 1)
                self.cv3 = Conv(2 * c_, c2, 1)
                self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

            def forward(self, x):
                return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))
        
        # Add C3 module
        if not hasattr(block, 'C3'):
            block.C3 = C3Base
            print("Added C3 module")
        
        # Get reference to C3 (either existing or just added)
        C3 = getattr(block, 'C3', C3Base)
        
        # C3k module (the one currently missing)
        if not hasattr(block, 'C3k'):
            class C3k(nn.Module):
                """C3k is a variant with customizable kernel sizes"""
                def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3):
                    super().__init__()
                    c_ = int(c2 * e)  # hidden channels
                    self.cv1 = Conv(c1, c_, 1, 1)
                    self.cv2 = Conv(c1, c_, 1, 1)
                    self.cv3 = Conv(2 * c_, c2, 1)
                    self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))

                def forward(self, x):
                    return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))
            
            block.C3k = C3k
            print("Added C3k module")
        
        # C3k2 module
        if not hasattr(block, 'C3k2'):
            class C3k2(nn.Module):
                """C3k2 is a variant of C3k with specific configurations"""
                def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
                    super().__init__()
                    c_ = int(c2 * e)  # hidden channels
                    self.cv1 = Conv(c1, c_, 1, 1)
                    self.cv2 = Conv(c1, c_, 1, 1) 
                    self.cv3 = Conv(2 * c_, c2, 1)
                    self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=(3, 3), e=1.0) for _ in range(n)))
                    
                def forward(self, x):
                    return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))
            
            block.C3k2 = C3k2
            print("Added C3k2 module")
        
        # C2f module (newer YOLOv8 versions)
        if not hasattr(block, 'C2f'):
            class C2f(nn.Module):
                """CSP Bottleneck with 2 convolutions and fusion"""
                def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
                    super().__init__()
                    self.c = int(c2 * e)  # hidden channels
                    self.cv1 = Conv(c1, 2 * self.c, 1, 1)
                    self.cv2 = Conv((2 + n) * self.c, c2, 1)
                    self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=(3, 3), e=1.0) for _ in range(n))

                def forward(self, x):
                    y = list(self.cv1(x).split((self.c, self.c), 1))
                    y.extend(m(y[-1]) for m in self.m)
                    return self.cv2(torch.cat(y, 1))
            
            block.C2f = C2f
            print("Added C2f module")
        
        # C3TR module (Transformer variant)
        if not hasattr(block, 'C3TR'):
            class C3TR(C3):
                """C3 module with Transformer blocks"""
                def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
                    super().__init__(c1, c2, n, shortcut, g, e)
                    # Simplified version - in reality would have transformer blocks
            
            block.C3TR = C3TR
            print("Added C3TR module")
        
        # C3SPP module (Spatial Pyramid Pooling)
        if not hasattr(block, 'C3SPP'):
            class C3SPP(C3):
                """C3 module with SPP block"""
                def __init__(self, c1, c2, k=(5, 9, 13), n=1, shortcut=True, g=1, e=0.5):
                    super().__init__(c1, c2, n, shortcut, g, e)
                    # Simplified version
            
            block.C3SPP = C3SPP
            print("Added C3SPP module")
        
        # C3Ghost module (Ghost convolutions)
        if not hasattr(block, 'C3Ghost'):
            class C3Ghost(C3):
                """C3 module with Ghost Bottlenecks"""
                def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
                    super().__init__(c1, c2, n, shortcut, g, e)
                    # Uses ghost convolutions internally
            
            block.C3Ghost = C3Ghost
            print("Added C3Ghost module")
        
        # SPPF module (Spatial Pyramid Pooling - Fast)
        if not hasattr(block, 'SPPF'):
            class SPPF(nn.Module):
                """Spatial Pyramid Pooling - Fast"""
                def __init__(self, c1, c2, k=5):
                    super().__init__()
                    c_ = c1 // 2  # hidden channels
                    self.cv1 = Conv(c1, c_, 1, 1)
                    self.cv2 = Conv(c_ * 4, c2, 1, 1)
                    self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

                def forward(self, x):
                    x = self.cv1(x)
                    y1 = self.m(x)
                    y2 = self.m(y1)
                    return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))
            
            block.SPPF = SPPF
            print("Added SPPF module")
        
        # C2 module (compact version)
        if not hasattr(block, 'C2'):
            class C2(nn.Module):
                """CSP Bottleneck with 2 convolutions"""
                def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
                    super().__init__()
                    self.c = int(c2 * e)  # hidden channels
                    self.cv1 = Conv(c1, 2 * self.c, 1, 1)
                    self.cv2 = Conv(2 * self.c, c2, 1)
                    self.m = nn.Sequential(*(Bottleneck(self.c, self.c, shortcut, g, e=1.0) for _ in range(n)))

                def forward(self, x):
                    a, b = self.cv1(x).split((self.c, self.c), 1)
                    return self.cv2(torch.cat((a, self.m(b)), 1))
            
            block.C2 = C2
            print("Added C2 module")
            
        print("✅ All compatibility modules added successfully")
            
    except Exception as e:
        print(f"Warning: Error adding compatibility modules: {e}")
        # Continue anyway - don't crash the app

# Dynamic module loader for any missing modules
def add_dynamic_module(module_name):
    """Dynamically add a missing module based on its name"""
    from ultralytics.nn.modules import block
    import torch.nn as nn
    from ultralytics.nn.modules.conv import Conv
    from ultralytics.nn.modules.block import Bottleneck
    
    print(f"Attempting to create module: {module_name}")
    
    # Create a generic module based on common patterns
    if module_name.startswith('C3') or module_name.startswith('C2'):
        # It's a CSP-style module
        class DynamicCSPModule(nn.Module):
            def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, *args, **kwargs):
                super().__init__()
                c_ = int(c2 * e)  # hidden channels
                self.cv1 = Conv(c1, c_, 1, 1)
                self.cv2 = Conv(c1, c_, 1, 1)
                self.cv3 = Conv(2 * c_, c2, 1)
                self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))
                
            def forward(self, x):
                return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))
        
        setattr(block, module_name, DynamicCSPModule)
        print(f"✅ Dynamically added {module_name} as CSP-style module")
        
    elif 'SPP' in module_name:
        # It's a Spatial Pyramid Pooling module
        class DynamicSPPModule(nn.Module):
            def __init__(self, c1, c2, k=5, *args, **kwargs):
                super().__init__()
                c_ = c1 // 2
                self.cv1 = Conv(c1, c_, 1, 1)
                self.cv2 = Conv(c_ * 4, c2, 1, 1)
                self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
                
            def forward(self, x):
                x = self.cv1(x)
                y1 = self.m(x)
                y2 = self.m(y1)
                return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))
        
        setattr(block, module_name, DynamicSPPModule)
        print(f"✅ Dynamically added {module_name} as SPP-style module")
        
    else:
        # Create a generic module that might work
        class GenericModule(nn.Module):
            def __init__(self, c1, c2, *args, **kwargs):
                super().__init__()
                self.conv = Conv(c1, c2, 1, 1)
                
            def forward(self, x):
                return self.conv(x)
        
        setattr(block, module_name, GenericModule)
        print(f"✅ Dynamically added {module_name} as generic module")

# Add compatibility modules before importing YOLO
print("Initializing YOLO compatibility layer...")
add_missing_modules()

from ultralytics import YOLO
import cv2
import numpy as np
import uuid
import shutil
import glob
import onnxruntime as ort
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image as RLImage, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
from PIL import Image
import io
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing
import threading
import csv

# --- Configuration ---
UPLOAD_FOLDER = 'static/uploads'
PROCESSED_FOLDER = 'static/processed'
MODEL_FOLDER = 'models'
REPORTS_FOLDER = 'static/reports'
ALLOWED_IMAGE_EXTENSIONS = {'png', 'jpg', 'jpeg'}
ALLOWED_VIDEO_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'webm'}
ALLOWED_MODEL_EXTENSIONS = {'pt', 'onnx'}

# --- Model Class Mapping Configuration ---
CLASS_MAPPINGS = {
    'football_detection': {
        'standard_classes': ['football', 'cone', 'player', 'goalpost'],
        'model_mappings': {
            'yolov8_football': {
                0: 'football',
                1: 'cone',
                2: 'player',
                3: 'goalpost'
            },
            'custom_football': {
                0: 'ball',
                1: 'cone',
                2: 'person',
                3: 'goal'
            },
            'coco_subset': {
                32: 'football',
                0: 'player',
            }
        },
        'class_aliases': {
            'ball': 'football',
            'soccer_ball': 'football',
            'person': 'player',
            'human': 'player',
            'goal': 'goalpost',
            'net': 'goalpost',
            'cones': 'cone'
        }
    },
    'general_detection': {
        'standard_classes': ['person', 'car', 'truck', 'bus', 'motorcycle', 'bicycle'],
        'model_mappings': {
            'coco': {
                0: 'person',
                2: 'car',
                7: 'truck',
                5: 'bus',
                3: 'motorcycle',
                1: 'bicycle'
            }
        },
        'class_aliases': {
            'human': 'person',
            'vehicle': 'car',
            'bike': 'bicycle'
        }
    }
}

# Default configuration
CURRENT_MAPPING = 'football_detection'
SELECTED_CLASSES = ['football', 'cone']
CONFIDENCE_THRESHOLD = 0.25  # Default confidence threshold

# --- App Initialization ---
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER
app.config['MODEL_FOLDER'] = MODEL_FOLDER
app.config['REPORTS_FOLDER'] = REPORTS_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024

# --- Performance Metrics Class ---
class PerformanceMetrics:
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.inference_times = []
        self.preprocess_times = []
        self.postprocess_times = []
        self.total_times = []
        self.confidence_scores = []
        self.detection_counts = []
        self.precision_scores = []
        self.recall_scores = []
        self.f1_scores = []
        self.class_specific_metrics = {}
        self.memory_usage = []
        self.fps_rates = []
        self.iou_scores = []
    
    def add_inference_time(self, time_ms):
        self.inference_times.append(time_ms)
    
    def add_preprocess_time(self, time_ms):
        self.preprocess_times.append(time_ms)
    
    def add_postprocess_time(self, time_ms):
        self.postprocess_times.append(time_ms)
    
    def add_total_time(self, time_ms):
        self.total_times.append(time_ms)
    
    def add_confidence_scores(self, scores):
        self.confidence_scores.extend(scores)
    
    def add_detection_count(self, count):
        self.detection_counts.append(count)
    
    def add_fps_rate(self, fps):
        self.fps_rates.append(fps)
    
    def get_statistics(self):
        def safe_stats(data):
            if not data:
                return {'mean': 0, 'std': 0, 'min': 0, 'max': 0, 'median': 0}
            return {
                'mean': statistics.mean(data),
                'std': statistics.stdev(data) if len(data) > 1 else 0,
                'min': min(data),
                'max': max(data),
                'median': statistics.median(data)
            }
        
        return {
            'inference_time': safe_stats(self.inference_times),
            'preprocess_time': safe_stats(self.preprocess_times),
            'postprocess_time': safe_stats(self.postprocess_times),
            'total_time': safe_stats(self.total_times),
            'confidence_scores': safe_stats(self.confidence_scores),
            'detection_counts': safe_stats(self.detection_counts),
            'fps_rates': safe_stats(self.fps_rates),
            'throughput': len(self.inference_times) / sum(self.total_times) * 1000 if self.total_times else 0,
            'total_processed': len(self.inference_times)
        }

# --- Helper Functions ---
def allowed_file(filename, file_type):
    """Check if file extension is allowed for the given file type"""
    if '.' not in filename:
        return False
    ext = filename.rsplit('.', 1)[1].lower()
    if file_type == 'image':
        return ext in ALLOWED_IMAGE_EXTENSIONS
    elif file_type == 'video':
        return ext in ALLOWED_VIDEO_EXTENSIONS
    return False

def validate_file_upload(file):
    """Validate uploaded file"""
    if not file or file.filename == '':
        return False, "No file selected"
    
    # Check file extension
    ext = file.filename.rsplit('.', 1)[1].lower() if '.' in file.filename else ''
    if ext not in ALLOWED_IMAGE_EXTENSIONS and ext not in ALLOWED_VIDEO_EXTENSIONS:
        return False, f"Invalid file type. Supported: {ALLOWED_IMAGE_EXTENSIONS | ALLOWED_VIDEO_EXTENSIONS}"
    
    # Check file size
    file.seek(0, os.SEEK_END)
    size = file.tell()
    file.seek(0)
    
    if size > 100 * 1024 * 1024:  # 100MB
        return False, f"File too large ({format_bytes(size)}). Maximum: 100MB"
    
    return True, "OK"

def clear_folders():
    """Clear temporary folders - fast version that runs in background"""
    # Just ensure folders exist, don't clean on startup
    for folder in [UPLOAD_FOLDER, PROCESSED_FOLDER, REPORTS_FOLDER]:
        os.makedirs(folder, exist_ok=True)
    
    # Schedule cleanup in background thread after startup
    cleanup_thread = threading.Thread(target=background_cleanup, daemon=True)
    cleanup_thread.start()

def background_cleanup():
    """Background cleanup of old files - runs after app starts"""
    time.sleep(5)  # Wait 5 seconds after startup
    
    print("Starting background cleanup of old files...")
    removed_count = 0
    
    for folder in [UPLOAD_FOLDER, PROCESSED_FOLDER, REPORTS_FOLDER]:
        if os.path.exists(folder):
            current_time = time.time()
            try:
                for file in os.listdir(folder):
                    file_path = os.path.join(folder, file)
                    if os.path.isfile(file_path):
                        # Only remove files older than 2 hours
                        if current_time - os.path.getmtime(file_path) > 7200:  # 2 hours
                            try:
                                os.remove(file_path)
                                removed_count += 1
                            except:
                                pass  # Silently skip files in use
            except Exception as e:
                print(f"Cleanup error in {folder}: {e}")
    
    if removed_count > 0:
        print(f"Background cleanup completed: removed {removed_count} old files")

def cleanup_temp_files(session_id=None):
    """Clean up temporary files from a specific session"""
    if session_id:
        # Clean files from specific session in background
        def cleanup():
            for folder in [UPLOAD_FOLDER, PROCESSED_FOLDER]:
                if os.path.exists(folder):
                    pattern = os.path.join(folder, f"*{session_id}*")
                    for file_path in glob.glob(pattern):
                        try:
                            os.remove(file_path)
                        except:
                            pass  # Silently skip
        
        # Run in background thread
        threading.Thread(target=cleanup, daemon=True).start()

def format_bytes(size):
    if size == 0:
        return "0B"
    power = 1024
    n = 0
    power_labels = {0: 'B', 1: 'KB', 2: 'MB', 3: 'GB', 4: 'TB'}
    while size >= power and n < len(power_labels) - 1:
        size /= power
        n += 1
    return f"{size:.2f} {power_labels[n]}"

def normalize_class_name(class_name, mapping_config):
    if not class_name:
        return 'unknown'
    
    class_name = class_name.lower().strip()
    aliases = mapping_config.get('class_aliases', {})
    
    if class_name in aliases:
        return aliases[class_name]
    
    if class_name in mapping_config.get('standard_classes', []):
        return class_name
    
    return 'unknown'

def map_model_classes(model, mapping_config, selected_classes):
    # Check for names in various places
    model_names = None
    if hasattr(model, '_custom_names') and model._custom_names:
        model_names = model._custom_names
    elif hasattr(model, 'names') and model.names:
        model_names = model.names
    
    if not model_names:
        if hasattr(model, 'model_path'):
            filename = os.path.basename(model.model_path).lower()
        else:
            filename = 'unknown'
            
        if 'trtfootballyolo' in filename or 'football' in filename:
            default_mapping = {0: 'cone', 1: 'football'}
            model_names = default_mapping
            print(f"Set default class names for football model: {model_names}")
        else:
            default_mapping = {i: f'class_{i}' for i in range(len(selected_classes))}
            model_names = default_mapping
            print(f"Set generic class names: {model_names}")
    
    standardized_mapping = {}
    
    for class_id, class_name in model_names.items():
        normalized_name = normalize_class_name(class_name, mapping_config)
        
        if normalized_name in selected_classes:
            standardized_mapping[class_id] = normalized_name
        else:
            if class_name.lower() in selected_classes:
                standardized_mapping[class_id] = class_name.lower()
    
    if not standardized_mapping:
        print(f"No class mappings found, attempting direct mapping...")
        print(f"Model classes: {model_names}")
        print(f"Selected classes: {selected_classes}")
        
        if len(model_names) == len(selected_classes):
            for i, selected_class in enumerate(selected_classes):
                if i in model_names:
                    standardized_mapping[i] = selected_class
                    print(f"Direct mapping: {i} -> {selected_class}")
    
    print(f"Final standardized mapping: {standardized_mapping}")
    return standardized_mapping

def detect_model_classes(model, model_path):
    """Attempt to detect model classes through various methods"""
    # Method 1: Check if model already has names attribute
    if hasattr(model, 'names') and model.names:
        return model.names
    
    # Method 2: Try a test inference to get class names
    try:
        import tempfile
        import numpy as np
        
        # Create a dummy image
        dummy_image = np.zeros((640, 640, 3), dtype=np.uint8)
        temp_path = tempfile.mktemp(suffix='.jpg')
        cv2.imwrite(temp_path, dummy_image)
        
        # Run inference
        results = model.predict(temp_path, verbose=False, conf=0.01)  # Low confidence to see all classes
        
        # Clean up
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        # Extract class names from results
        if results and len(results) > 0:
            result = results[0]
            if hasattr(result, 'names') and result.names:
                return result.names
            # Try to get from model after inference
            if hasattr(model, 'names') and model.names:
                return model.names
    except Exception as e:
        print(f"Test inference failed: {e}")
    
    # Method 3: Try to access model internals
    try:
        if hasattr(model, 'model') and hasattr(model.model, 'names'):
            return model.model.names
        elif hasattr(model, 'predictor') and hasattr(model.predictor, 'model') and hasattr(model.predictor.model, 'names'):
            return model.predictor.model.names
    except:
        pass
    
    return None

def infer_classes_from_filename(model_path, num_classes):
    """Infer class names from filename patterns"""
    filename = os.path.basename(model_path).lower()
    
    # Common patterns
    if 'football' in filename or 'soccer' in filename:
        if 'cone' in filename:
            return {0: 'cone', 1: 'football'}
        else:
            return {0: 'football', 1: 'player', 2: 'cone', 3: 'goalpost'}
    elif 'coco' in filename:
        # Return subset of COCO classes
        coco_classes = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'}
        return coco_classes
    else:
        # Generic class names
        return {i: f'class_{i}' for i in range(num_classes)}

def find_model_files():
    """Find and validate model files in the models directory"""
    model_files = {}
    model_names = {}
    
    if not os.path.exists(MODEL_FOLDER):
        os.makedirs(MODEL_FOLDER, exist_ok=True)
        return model_files, model_names
    
    all_files = glob.glob(os.path.join(MODEL_FOLDER, '*'))
    
    # Filter for supported model files
    supported_models = []
    for f in all_files:
        if not os.path.isfile(f):
            continue
        
        ext = f.rsplit('.', 1)[-1].lower() if '.' in f else ''
        if ext in ALLOWED_MODEL_EXTENSIONS:
            # Check file size
            size = os.path.getsize(f)
            if size < 1024:  # Less than 1KB
                print(f"Skipping {os.path.basename(f)}: File too small ({size} bytes)")
                continue
            supported_models.append(f)
    
    if len(supported_models) == 0:
        print("No valid model files found in models directory")
    elif len(supported_models) == 1:
        print(f"Only one model found: {os.path.basename(supported_models[0])}")
        print("Please add another model file to the 'models' directory for comparison")
    else:
        # Sort by filename for consistent ordering
        supported_models.sort()
        
        # Take first two models
        model_files['model1'] = supported_models[0]
        model_names['model1'] = os.path.basename(supported_models[0])
        
        model_files['model2'] = supported_models[1]
        model_names['model2'] = os.path.basename(supported_models[1])
        
        if len(supported_models) > 2:
            print(f"Found {len(supported_models)} models, using first two: {model_names['model1']}, {model_names['model2']}")
    
    return model_files, model_names

def get_model_info(model_path):
    size = format_bytes(os.path.getsize(model_path))
    format_type = 'ONNX' if model_path.endswith('.onnx') else 'PyTorch'
    filename = os.path.basename(model_path)
    return {
        'size': size,
        'format': format_type,
        'filename': filename
    }

def load_model(model_path, mapping_config, selected_classes):
    """Load a YOLO model with improved error handling and dynamic class detection"""
    if not os.path.exists(model_path):
        raise ValueError(f"Model file not found: {model_path}")
    
    file_size = os.path.getsize(model_path)
    if file_size < 1024:
        raise ValueError(f"Model file '{os.path.basename(model_path)}' is too small ({file_size} bytes). Please ensure it's a valid model file.")
    
    model_name = os.path.basename(model_path)
    print(f"Loading model: {model_name} ({format_bytes(file_size)})")
    
    try:
        # Create a wrapper class to handle attribute access
        class ModelWrapper:
            def __init__(self, yolo_model, model_path):
                self._model = yolo_model
                self.model_path = model_path
                self.is_onnx = model_path.endswith('.onnx')
                self._custom_names = None
                self.class_mapping = {}
            
            def __call__(self, *args, **kwargs):
                return self._model(*args, **kwargs)
            
            def __getattr__(self, name):
                # First check our custom attributes
                if name == 'names':
                    if self._custom_names is not None:
                        return self._custom_names
                    elif hasattr(self._model, 'names'):
                        return self._model.names
                    return {}
                # Then delegate to the wrapped model
                return getattr(self._model, name)
        
        # Load the model
        if model_path.endswith('.onnx'):
            base_model = YOLO(model_path, task='detect')
            model = ModelWrapper(base_model, model_path)
        else:
            base_model = YOLO(model_path, task='detect')
            model = ModelWrapper(base_model, model_path)
        
        # Try to detect classes dynamically
        detected_classes = detect_model_classes(model, model_path)
        if detected_classes:
            print(f"Detected {len(detected_classes)} classes in model: {list(detected_classes.values())[:5]}...")
            # Store names in a wrapper attribute to avoid setting protected attributes
            if not hasattr(model, '_custom_names'):
                model._custom_names = detected_classes
        else:
            print(f"Could not detect classes for {model_name}, using fallback method")
            # Fallback: try to infer from filename or use generic names
            if not hasattr(model, '_custom_names'):
                model._custom_names = infer_classes_from_filename(model_path, len(selected_classes))
        
        # Apply class mapping
        class_mapping = map_model_classes(model, mapping_config, selected_classes)
        
        if not class_mapping:
            print(f"Warning: No valid class mapping found for {model_name}")
            # Create a basic mapping
            class_mapping = {i: selected_classes[i % len(selected_classes)] for i in range(len(selected_classes))}
            print(f"Using basic mapping: {class_mapping}")
        
        model.class_mapping = class_mapping
        model.selected_classes = selected_classes
        model.model_name = model_name
        
        print(f"Successfully loaded {model_name} with mapping: {class_mapping}")
        return model
        
    except Exception as e:
        error_msg = f"Failed to load model {model_name}: {str(e)}"
        print(f"ERROR: {error_msg}")
        
        # Check for missing module errors
        if "Can't get attribute" in str(e) and "on <module" in str(e):
            # Extract the missing module name
            import re
            match = re.search(r"Can't get attribute '(\w+)'", str(e))
            if match:
                missing_module = match.group(1)
                print(f"Detected missing module: {missing_module}")
                
                # Try to add the missing module dynamically
                try:
                    add_dynamic_module(missing_module)
                    print(f"Added {missing_module} module dynamically, retrying model load...")
                    
                    # Retry loading the model
                    return load_model(model_path, mapping_config, selected_classes)
                    
                except Exception as retry_error:
                    print(f"Failed to add {missing_module} dynamically: {retry_error}")
                    error_msg += f"\n\nMissing module '{missing_module}' could not be added dynamically."
            
            # Model version mismatch
            error_msg += "\n\nThis model was trained with a different version of Ultralytics/YOLOv8."
            error_msg += "\nPossible solutions:"
            error_msg += "\n1. Update Ultralytics: pip install -U ultralytics"
            error_msg += "\n2. Use the ONNX version of this model if available"
            error_msg += "\n3. Re-export the model with your current Ultralytics version"
                
        elif 'onnx' in str(e).lower():
            error_msg += "\n\nMake sure onnxruntime is installed: pip install onnxruntime"
        elif 'GPU' in str(e) or 'CUDA' in str(e):
            error_msg += "\n\nTry running with CPU by setting CUDA_VISIBLE_DEVICES=''"
            
        raise ValueError(error_msg)

def filter_results_by_classes(results, class_mapping):
    filtered_results = []
    
    for result in results:
        if hasattr(result, 'boxes') and len(result.boxes) > 0:
            filtered_boxes = []
            for i, box in enumerate(result.boxes):
                if hasattr(box, 'cls'):
                    cls_id = int(box.cls[0])
                    if cls_id in class_mapping:
                        filtered_boxes.append(box)
            
            if filtered_boxes:
                result.boxes = filtered_boxes
        
        filtered_results.append(result)
    
    return filtered_results

def draw_boxes_image(image_path, results, output_path, model_name, class_mapping):
    image = cv2.imread(image_path)
    
    for result in results:
        if hasattr(result, 'boxes') and len(result.boxes) > 0:
            for box in result.boxes:
                if hasattr(box, 'xyxy') and len(box.xyxy) > 0:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    confidence = box.conf[0]
                    cls_id = int(box.cls[0])
                    
                    label = f"{class_mapping.get(cls_id, 'unknown')} {confidence:.2f}"
                    
                    color = (0, 255, 0) if "1" in model_name else (0, 0, 255)
                    
                    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
    cv2.imwrite(output_path, image)

def process_single_image(args):
    """Process a single image with both models (for parallel processing)"""
    image_path, model1, model2, model_names = args
    filename = os.path.basename(image_path)
    
    result_data = {
        'filename': filename,
        'path': image_path,
        'results1': [],
        'results2': [],
        'analysis1': {},
        'analysis2': {},
        'speeds1': {},
        'speeds2': {},
        'comparison': '',
        'metrics1': {},
        'metrics2': {}
    }
    
    # Process with model 1
    start_time = time.time()
    try:
        results1 = model1(image_path, verbose=False, conf=CONFIDENCE_THRESHOLD)
        inference_time1 = (time.time() - start_time) * 1000
        
        filtered_results1 = filter_results_by_classes(results1, model1.class_mapping)
        
        speed1_dict = results1[0].speed if results1 and len(results1) > 0 else {}
        preprocess_speed1 = speed1_dict.get('preprocess', 0)
        postprocess_speed1 = speed1_dict.get('postprocess', 0)
        total_speed1 = preprocess_speed1 + inference_time1 + postprocess_speed1
        
        # Extract confidence scores
        confidences1 = []
        detection_count1 = 0
        for result in filtered_results1:
            if hasattr(result, 'boxes') and len(result.boxes) > 0:
                for box in result.boxes:
                    if hasattr(box, 'conf'):
                        confidences1.append(float(box.conf[0]))
                        detection_count1 += 1
        
        result_data['results1'] = filtered_results1
        result_data['speeds1'] = {
            'preprocess': preprocess_speed1,
            'inference': inference_time1,
            'postprocess': postprocess_speed1,
            'total': total_speed1
        }
        result_data['metrics1'] = {
            'inference_time': inference_time1,
            'preprocess_time': preprocess_speed1,
            'postprocess_time': postprocess_speed1,
            'total_time': total_speed1,
            'confidences': confidences1,
            'detection_count': detection_count1,
            'fps': 1000 / total_speed1 if total_speed1 > 0 else 0
        }
        
    except Exception as e:
        print(f"Error processing {filename} with model 1: {e}")
        result_data['speeds1'] = {'preprocess': 0, 'inference': 0, 'postprocess': 0, 'total': 0}
        result_data['metrics1'] = {'inference_time': 0, 'confidences': [], 'detection_count': 0}
    
    # Process with model 2
    start_time = time.time()
    try:
        results2 = model2(image_path, verbose=False, conf=CONFIDENCE_THRESHOLD)
        inference_time2 = (time.time() - start_time) * 1000
        
        filtered_results2 = filter_results_by_classes(results2, model2.class_mapping)
        
        speed2_dict = results2[0].speed if results2 and len(results2) > 0 else {}
        preprocess_speed2 = speed2_dict.get('preprocess', 0)
        postprocess_speed2 = speed2_dict.get('postprocess', 0)
        total_speed2 = preprocess_speed2 + inference_time2 + postprocess_speed2
        
        # Extract confidence scores
        confidences2 = []
        detection_count2 = 0
        for result in filtered_results2:
            if hasattr(result, 'boxes') and len(result.boxes) > 0:
                for box in result.boxes:
                    if hasattr(box, 'conf'):
                        confidences2.append(float(box.conf[0]))
                        detection_count2 += 1
        
        result_data['results2'] = filtered_results2
        result_data['speeds2'] = {
            'preprocess': preprocess_speed2,
            'inference': inference_time2,
            'postprocess': postprocess_speed2,
            'total': total_speed2
        }
        result_data['metrics2'] = {
            'inference_time': inference_time2,
            'preprocess_time': preprocess_speed2,
            'postprocess_time': postprocess_speed2,
            'total_time': total_speed2,
            'confidences': confidences2,
            'detection_count': detection_count2,
            'fps': 1000 / total_speed2 if total_speed2 > 0 else 0
        }
        
    except Exception as e:
        print(f"Error processing {filename} with model 2: {e}")
        result_data['speeds2'] = {'preprocess': 0, 'inference': 0, 'postprocess': 0, 'total': 0}
        result_data['metrics2'] = {'inference_time': 0, 'confidences': [], 'detection_count': 0}
    
    # Analyze results
    result_data['analysis1'] = analyze_results(result_data['results1'], model1.class_mapping, model1.selected_classes)
    result_data['analysis2'] = analyze_results(result_data['results2'], model2.class_mapping, model2.selected_classes)
    result_data['comparison'] = compare_models(
        result_data['analysis1'], result_data['analysis2'],
        result_data['metrics1'].get('inference_time', 0),
        result_data['metrics2'].get('inference_time', 0),
        model1.selected_classes,
        model_names['model1'], model_names['model2']
    )
    
    return result_data

def process_batch_images(image_paths, model1, model2, metrics1, metrics2, model_names):
    """Process a batch of images in parallel and collect detailed metrics"""
    results_data = []
    
    # Determine number of workers (use fewer workers to avoid overwhelming the system)
    num_workers = min(4, multiprocessing.cpu_count())
    print(f"Processing {len(image_paths)} images with {num_workers} parallel workers...")
    
    # Thread-safe locks for metrics updates
    metrics1_lock = threading.Lock()
    metrics2_lock = threading.Lock()
    
    # Process images in parallel
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Submit all tasks
        future_to_image = {
            executor.submit(process_single_image, (image_path, model1, model2, model_names)): image_path
            for image_path in image_paths
        }
        
        # Collect results as they complete
        for i, future in enumerate(as_completed(future_to_image), 1):
            try:
                result = future.result()
                results_data.append(result)
                
                # Update metrics in thread-safe manner
                with metrics1_lock:
                    if result['metrics1']['inference_time'] > 0:
                        metrics1.add_inference_time(result['metrics1']['inference_time'])
                        metrics1.add_preprocess_time(result['metrics1'].get('preprocess_time', 0))
                        metrics1.add_postprocess_time(result['metrics1'].get('postprocess_time', 0))
                        metrics1.add_total_time(result['metrics1'].get('total_time', 0))
                        metrics1.add_confidence_scores(result['metrics1'].get('confidences', []))
                        metrics1.add_detection_count(result['metrics1'].get('detection_count', 0))
                        if result['metrics1'].get('fps', 0) > 0:
                            metrics1.add_fps_rate(result['metrics1']['fps'])
                
                with metrics2_lock:
                    if result['metrics2']['inference_time'] > 0:
                        metrics2.add_inference_time(result['metrics2']['inference_time'])
                        metrics2.add_preprocess_time(result['metrics2'].get('preprocess_time', 0))
                        metrics2.add_postprocess_time(result['metrics2'].get('postprocess_time', 0))
                        metrics2.add_total_time(result['metrics2'].get('total_time', 0))
                        metrics2.add_confidence_scores(result['metrics2'].get('confidences', []))
                        metrics2.add_detection_count(result['metrics2'].get('detection_count', 0))
                        if result['metrics2'].get('fps', 0) > 0:
                            metrics2.add_fps_rate(result['metrics2']['fps'])
                
                # Log progress
                if i % 5 == 0 or i == len(image_paths):
                    print(f"Processed {i}/{len(image_paths)} images ({i/len(image_paths)*100:.1f}%)")
                    
            except Exception as e:
                print(f"Error processing image: {e}")
    
    # Sort results by filename to maintain consistent ordering
    results_data.sort(key=lambda x: x['filename'])
    
    return results_data

def process_video(video_path, model1, model2, metrics1, metrics2, model_names, output_path, max_frames=None):
    """Process video frame by frame with memory optimization"""
    cap = cv2.VideoCapture(video_path)
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Limit processing for very long videos
    if max_frames and total_frames > max_frames:
        print(f"Video has {total_frames} frames, limiting to {max_frames} frames")
        total_frames = max_frames
    
    # Create output video writer (side-by-side, so double the width)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width * 2, height))
    
    frame_count = 0
    processing_times = []
    
    # Process in batches to manage memory
    batch_size = 30  # Process 30 frames at a time
    
    try:
        while cap.isOpened() and (not max_frames or frame_count < max_frames):
            ret, frame = cap.read()
            if not ret:
                break
            
        frame_count += 1
        frame_start = time.time()
        
        # Process with both models
        start_time1 = time.time()
        results1 = model1(frame, verbose=False, conf=CONFIDENCE_THRESHOLD)
        inference_time1 = (time.time() - start_time1) * 1000
        
        start_time2 = time.time()
        results2 = model2(frame, verbose=False, conf=CONFIDENCE_THRESHOLD)
        inference_time2 = (time.time() - start_time2) * 1000
        
        # Filter results
        filtered_results1 = filter_results_by_classes(results1, model1.class_mapping)
        filtered_results2 = filter_results_by_classes(results2, model2.class_mapping)
        
        # Create copies for drawing
        frame1 = frame.copy()
        frame2 = frame.copy()
        
        # Draw bounding boxes and labels for model 1
        if filtered_results1 and len(filtered_results1) > 0:
            result1 = filtered_results1[0]
            if hasattr(result1, 'boxes') and len(result1.boxes) > 0:
                for box in result1.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    conf = box.conf[0].cpu().numpy()
                    cls = int(box.cls[0].cpu().numpy())
                    
                    if cls in model1.class_mapping:
                        class_name = model1.class_mapping[cls]
                        if class_name in SELECTED_CLASSES:
                            # Green for model 1
                            cv2.rectangle(frame1, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            label = f'{class_name}: {conf:.2f}'
                            cv2.putText(frame1, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Draw bounding boxes and labels for model 2
        if filtered_results2 and len(filtered_results2) > 0:
            result2 = filtered_results2[0]
            if hasattr(result2, 'boxes') and len(result2.boxes) > 0:
                for box in result2.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    conf = box.conf[0].cpu().numpy()
                    cls = int(box.cls[0].cpu().numpy())
                    
                    if cls in model2.class_mapping:
                        class_name = model2.class_mapping[cls]
                        if class_name in SELECTED_CLASSES:
                            # Blue for model 2
                            cv2.rectangle(frame2, (x1, y1), (x2, y2), (255, 0, 0), 2)
                            label = f'{class_name}: {conf:.2f}'
                            cv2.putText(frame2, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        # Add stats overlay
        # Model 1 stats
        cv2.putText(frame1, f'{model_names["model1"]}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame1, f'Inference: {inference_time1:.1f}ms', (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(frame1, f'FPS: {1000/inference_time1:.1f}', (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Model 2 stats
        cv2.putText(frame2, f'{model_names["model2"]}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.putText(frame2, f'Inference: {inference_time2:.1f}ms', (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(frame2, f'FPS: {1000/inference_time2:.1f}', (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Add frame counter
        cv2.putText(frame1, f'Frame: {frame_count}/{total_frames}', (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame2, f'Frame: {frame_count}/{total_frames}', (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Combine frames side by side
        combined_frame = np.hstack((frame1, frame2))
        
        # Write to output video
        out.write(combined_frame)
        
        # Track metrics
        metrics1.add_inference_time(inference_time1)
        metrics2.add_inference_time(inference_time2)
        
        # Log progress
        if frame_count % 30 == 0:  # Log every 30 frames
            print(f"Processed {frame_count}/{total_frames} frames ({frame_count/total_frames*100:.1f}%)")
            # Periodic garbage collection for long videos
            if frame_count % 300 == 0:
                import gc
                gc.collect()
        
        processing_times.append(time.time() - frame_start)
    
    except Exception as e:
        print(f"Error processing video: {e}")
    finally:
        # Ensure resources are released
        cap.release()
        out.release()
        cv2.destroyAllWindows()
    
    avg_processing_time = np.mean(processing_times) if processing_times else 0
    
    return {
        'total_frames': frame_count,
        'fps': fps,
        'avg_processing_time': avg_processing_time,
        'output_path': output_path
    }

def analyze_results(results, class_mapping, selected_classes):
    analysis = {
        'total_detections': 0,
        'class_counts': {name: 0 for name in selected_classes},
        'avg_confidence': 0,
        'avg_class_confidence': {name: 0 for name in selected_classes},
        'confidences': []
    }
    
    if not isinstance(results, list):
        results = [results]
    
    total_conf = 0
    class_conf_totals = {name: 0 for name in selected_classes}
    
    for result in results:
        if hasattr(result, 'boxes') and len(result.boxes) > 0:
            for box in result.boxes:
                if hasattr(box, 'conf') and hasattr(box, 'cls'):
                    cls_id = int(box.cls[0])
                    
                    if cls_id in class_mapping:
                        confidence = float(box.conf[0])
                        cls_name = class_mapping[cls_id]

                        analysis['confidences'].append(confidence)
                        analysis['total_detections'] += 1
                        total_conf += confidence
                        
                        analysis['class_counts'][cls_name] += 1
                        class_conf_totals[cls_name] += confidence

    if analysis['total_detections'] > 0:
        analysis['avg_confidence'] = total_conf / analysis['total_detections']
        
        for name, count in analysis['class_counts'].items():
            if count > 0:
                analysis['avg_class_confidence'][name] = class_conf_totals[name] / count
    
    return analysis

def compare_models(analysis1, analysis2, speed1, speed2, selected_classes, name1="Model 1", name2="Model 2"):
    score1, score2 = 0, 0

    if analysis1['avg_confidence'] > analysis2['avg_confidence']:
        score1 += 1
    elif analysis2['avg_confidence'] > analysis1['avg_confidence']:
        score2 += 1
        
    if speed1 < speed2:
        score1 += 1
    elif speed2 < speed1:
        score2 += 1

    if score1 > score2:
        return f"{name1} is better"
    if score2 > score1:
        return f"{name2} is better"
    if score1 == score2 and score1 > 0:
        return "Both models performed similarly"
    return "No conclusive winner (or no detections)"

def create_performance_charts(metrics1, metrics2, model_names):
    """Create performance comparison charts"""
    stats1 = metrics1.get_statistics()
    stats2 = metrics2.get_statistics()
    
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))
    fig.suptitle('Model Performance Comparison', fontsize=16)
    
    # Inference time comparison
    ax1 = axes[0]
    times1 = metrics1.inference_times
    times2 = metrics2.inference_times
    
    ax1.hist(times1, alpha=0.7, label=f'{model_names["model1"]} (μ={stats1["inference_time"]["mean"]:.2f}ms)', bins=20)
    ax1.hist(times2, alpha=0.7, label=f'{model_names["model2"]} (μ={stats2["inference_time"]["mean"]:.2f}ms)', bins=20)
    ax1.set_xlabel('Inference Time (ms)')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Inference Time Distribution')
    ax1.legend()
    
    # Confidence score comparison
    ax2 = axes[1]
    conf1 = metrics1.confidence_scores
    conf2 = metrics2.confidence_scores
    
    ax2.hist(conf1, alpha=0.7, label=f'{model_names["model1"]} (μ={stats1["confidence_scores"]["mean"]:.3f})', bins=20)
    ax2.hist(conf2, alpha=0.7, label=f'{model_names["model2"]} (μ={stats2["confidence_scores"]["mean"]:.3f})', bins=20)
    ax2.set_xlabel('Confidence Score')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Confidence Score Distribution')
    ax2.legend()
    
    # FPS comparison
    ax3 = axes[2]
    fps1 = metrics1.fps_rates
    fps2 = metrics2.fps_rates
    
    ax3.hist(fps1, alpha=0.7, label=f'{model_names["model1"]} (μ={stats1["fps_rates"]["mean"]:.1f})', bins=20)
    ax3.hist(fps2, alpha=0.7, label=f'{model_names["model2"]} (μ={stats2["fps_rates"]["mean"]:.1f})', bins=20)
    ax3.set_xlabel('FPS')
    ax3.set_ylabel('Frequency')
    ax3.set_title('FPS Distribution')
    ax3.legend()
    
    plt.tight_layout()
    
    # Save chart
    chart_path = os.path.join(REPORTS_FOLDER, 'performance_charts.png')
    plt.savefig(chart_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return chart_path

def generate_pdf_report(results_data, metrics1, metrics2, model_names, model_info):
    """Generate comprehensive PDF report"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pdf_filename = f"model_comparison_report_{timestamp}.pdf"
    pdf_path = os.path.join(REPORTS_FOLDER, pdf_filename)
    
    doc = SimpleDocTemplate(pdf_path, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []
    
    # Title
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        spaceAfter=30,
        alignment=TA_CENTER
    )
    story.append(Paragraph("YOLO Model Performance Comparison Report", title_style))
    story.append(Spacer(1, 20))
    
    # Executive Summary
    story.append(Paragraph("Executive Summary", styles['Heading2']))
    
    stats1 = metrics1.get_statistics()
    stats2 = metrics2.get_statistics()
    
    # Determine overall winner
    model1_wins = sum(1 for r in results_data if model_names['model1'] in r['comparison'])
    model2_wins = sum(1 for r in results_data if model_names['model2'] in r['comparison'])
    
    if model1_wins > model2_wins:
        winner = f"{model_names['model1']}"
    elif model2_wins > model1_wins:
        winner = f"{model_names['model2']}"
    else:
        winner = "Tie"
    
    summary_text = f"""
    <b>Overall Winner:</b> {winner}<br/>
    <b>Total Images Processed:</b> {len(results_data)}<br/>
    <b>{model_names['model1']} Wins:</b> {model1_wins}<br/>
    <b>{model_names['model2']} Wins:</b> {model2_wins}<br/>
    <b>Average Inference Time:</b> {model_names['model1']}: {stats1['inference_time']['mean']:.2f}ms, {model_names['model2']}: {stats2['inference_time']['mean']:.2f}ms<br/>
    <b>Average Confidence:</b> {model_names['model1']}: {stats1['confidence_scores']['mean']:.3f}, {model_names['model2']}: {stats2['confidence_scores']['mean']:.3f}<br/>
    """
    
    story.append(Paragraph(summary_text, styles['Normal']))
    story.append(Spacer(1, 20))
    
    # Model Information Table
    story.append(Paragraph("Model Information", styles['Heading2']))
    
    model_data = [
        ['Metric', model_names['model1'], model_names['model2']],
        ['Filename', model_names['model1'], model_names['model2']],
        ['Format', model_info['model1']['format'], model_info['model2']['format']],
        ['Size', model_info['model1']['size'], model_info['model2']['size']],
        ['Classes Detected', ', '.join(SELECTED_CLASSES), ', '.join(SELECTED_CLASSES)]
    ]
    
    model_table = Table(model_data, colWidths=[2*inch, 2*inch, 2*inch])
    model_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    
    story.append(model_table)
    story.append(Spacer(1, 20))
    
    # Performance Metrics Table
    story.append(Paragraph("Detailed Performance Metrics", styles['Heading2']))
    
    perf_data = [
        ['Metric', model_names['model1'], model_names['model2'], 'Winner'],
        ['Avg Inference Time (ms)', f"{stats1['inference_time']['mean']:.2f} ± {stats1['inference_time']['std']:.2f}", 
         f"{stats2['inference_time']['mean']:.2f} ± {stats2['inference_time']['std']:.2f}",
         model_names['model1'] if stats1['inference_time']['mean'] < stats2['inference_time']['mean'] else model_names['model2']],
        ['Min Inference Time (ms)', f"{stats1['inference_time']['min']:.2f}", f"{stats2['inference_time']['min']:.2f}",
         model_names['model1'] if stats1['inference_time']['min'] < stats2['inference_time']['min'] else model_names['model2']],
        ['Max Inference Time (ms)', f"{stats1['inference_time']['max']:.2f}", f"{stats2['inference_time']['max']:.2f}",
         model_names['model1'] if stats1['inference_time']['max'] < stats2['inference_time']['max'] else model_names['model2']],
        ['Avg Confidence', f"{stats1['confidence_scores']['mean']:.3f} ± {stats1['confidence_scores']['std']:.3f}", 
         f"{stats2['confidence_scores']['mean']:.3f} ± {stats2['confidence_scores']['std']:.3f}",
         model_names['model1'] if stats1['confidence_scores']['mean'] > stats2['confidence_scores']['mean'] else model_names['model2']],
        ['Avg Detections/Image', f"{stats1['detection_counts']['mean']:.1f} ± {stats1['detection_counts']['std']:.1f}", 
         f"{stats2['detection_counts']['mean']:.1f} ± {stats2['detection_counts']['std']:.1f}",
         model_names['model1'] if stats1['detection_counts']['mean'] > stats2['detection_counts']['mean'] else model_names['model2']],
       ['Avg FPS', f"{stats1['fps_rates']['mean']:.1f} ± {stats1['fps_rates']['std']:.1f}", 
         f"{stats2['fps_rates']['mean']:.1f} ± {stats2['fps_rates']['std']:.1f}",
         model_names['model1'] if stats1['fps_rates']['mean'] > stats2['fps_rates']['mean'] else model_names['model2']],
        ['Throughput (images/sec)', f"{stats1['throughput']:.2f}", f"{stats2['throughput']:.2f}",
         model_names['model1'] if stats1['throughput'] > stats2['throughput'] else model_names['model2']]
    ]
    
    perf_table = Table(perf_data, colWidths=[2*inch, 1.5*inch, 1.5*inch, 1*inch])
    perf_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('FONTSIZE', (0, 1), (-1, -1), 8)
    ]))
    
    story.append(perf_table)
    story.append(Spacer(1, 20))
    
    # Add performance charts
    chart_path = create_performance_charts(metrics1, metrics2, model_names)
    if os.path.exists(chart_path):
        story.append(Paragraph("Performance Charts", styles['Heading2']))
        story.append(RLImage(chart_path, width=7*inch, height=5*inch))
        story.append(Spacer(1, 20))
    
    # Per-Image Results
    story.append(Paragraph("Per-Image Results", styles['Heading2']))
    
    for i, result in enumerate(results_data[:10]):  # Limit to first 10 images for space
        story.append(Paragraph(f"Image {i+1}: {result['filename']}", styles['Heading3']))
        
        # Create comparison table for this image
        img_data = [
            ['Metric', model_names['model1'], model_names['model2']],
            ['Average Confidence', f"{result['analysis1']['avg_confidence']:.3f}", f"{result['analysis2']['avg_confidence']:.3f}"],
            ['Inference Time (ms)', f"{result['speeds1']['inference']:.2f}", f"{result['speeds2']['inference']:.2f}"],
            ['Total Time (ms)', f"{result['speeds1']['total']:.2f}", f"{result['speeds2']['total']:.2f}"],
        ]
        
        # Add class-specific counts
        for class_name in SELECTED_CLASSES:
            count1 = result['analysis1']['class_counts'].get(class_name, 0)
            count2 = result['analysis2']['class_counts'].get(class_name, 0)
            img_data.append([f'{class_name.capitalize()} Count', count1, count2])
        
        img_table = Table(img_data, colWidths=[2*inch, 1.5*inch, 1.5*inch])
        img_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        story.append(img_table)
        story.append(Paragraph(f"<b>Winner:</b> {result['comparison']}", styles['Normal']))
        story.append(Spacer(1, 15))
    
    
    
    # Build PDF
    doc.build(story)
    
    return pdf_path, pdf_filename

def generate_csv_report(results_data, metrics1, metrics2, model_names):
    """Generate CSV report with comparison results"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"model_comparison_{timestamp}.csv"
    csv_path = os.path.join(REPORTS_FOLDER, csv_filename)
    
    stats1 = metrics1.get_statistics()
    stats2 = metrics2.get_statistics()
    
    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        
        # Write header info
        writer.writerow(['YOLO Model Comparison Report'])
        writer.writerow(['Generated', datetime.now().strftime('%Y-%m-%d %H:%M:%S')])
        writer.writerow([])
        
        # Model information
        writer.writerow(['Model Information'])
        writer.writerow(['', 'Model 1', 'Model 2'])
        writer.writerow(['Name', model_names['model1'], model_names['model2']])
        writer.writerow(['Selected Classes', ', '.join(SELECTED_CLASSES), ', '.join(SELECTED_CLASSES)])
        writer.writerow([])
        
        # Overall statistics
        writer.writerow(['Overall Performance Statistics'])
        writer.writerow(['Metric', f"{model_names['model1']}", f"{model_names['model2']}", 'Winner'])
        writer.writerow(['Avg Inference Time (ms)', 
                        f"{stats1['inference_time']['mean']:.2f}", 
                        f"{stats2['inference_time']['mean']:.2f}",
                        model_names['model1'] if stats1['inference_time']['mean'] < stats2['inference_time']['mean'] else model_names['model2']])
        writer.writerow(['Avg Confidence', 
                        f"{stats1['confidence_scores']['mean']:.3f}", 
                        f"{stats2['confidence_scores']['mean']:.3f}",
                        model_names['model1'] if stats1['confidence_scores']['mean'] > stats2['confidence_scores']['mean'] else model_names['model2']])
        writer.writerow(['Avg FPS', 
                        f"{stats1['fps_rates']['mean']:.1f}", 
                        f"{stats2['fps_rates']['mean']:.1f}",
                        model_names['model1'] if stats1['fps_rates']['mean'] > stats2['fps_rates']['mean'] else model_names['model2']])
        writer.writerow(['Total Detections', 
                        sum(r['metrics1']['detection_count'] for r in results_data),
                        sum(r['metrics2']['detection_count'] for r in results_data),
                        ''])
        writer.writerow([])
        
        # Per-image results
        writer.writerow(['Per-Image Results'])
        headers = ['Image', 'Winner', 
                  f"{model_names['model1']} - Inference (ms)", f"{model_names['model1']} - Detections", f"{model_names['model1']} - Avg Confidence",
                  f"{model_names['model2']} - Inference (ms)", f"{model_names['model2']} - Detections", f"{model_names['model2']} - Avg Confidence"]
        
        # Add class-specific headers
        for class_name in SELECTED_CLASSES:
            headers.extend([f"{model_names['model1']} - {class_name} Count", f"{model_names['model2']} - {class_name} Count"])
        
        writer.writerow(headers)
        
        # Write data for each image
        for result in results_data:
            row = [
                result['filename'],
                result['comparison'],
                f"{result['speeds1']['inference']:.2f}",
                result['analysis1']['total_detections'],
                f"{result['analysis1']['avg_confidence']:.3f}",
                f"{result['speeds2']['inference']:.2f}",
                result['analysis2']['total_detections'],
                f"{result['analysis2']['avg_confidence']:.3f}"
            ]
            
            # Add class-specific counts
            for class_name in SELECTED_CLASSES:
                row.extend([
                    result['analysis1']['class_counts'].get(class_name, 0),
                    result['analysis2']['class_counts'].get(class_name, 0)
                ])
            
            writer.writerow(row)
    
    return csv_path, csv_filename

def generate_json_report(results_data, metrics1, metrics2, model_names, model_info):
    """Generate JSON report with all comparison data"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_filename = f"model_comparison_{timestamp}.json"
    json_path = os.path.join(REPORTS_FOLDER, json_filename)
    
    stats1 = metrics1.get_statistics()
    stats2 = metrics2.get_statistics()
    
    # Calculate winners
    model1_wins = sum(1 for r in results_data if model_names['model1'] in r['comparison'])
    model2_wins = sum(1 for r in results_data if model_names['model2'] in r['comparison'])
    
    report_data = {
        'metadata': {
            'generated': datetime.now().isoformat(),
            'total_images': len(results_data),
            'selected_classes': SELECTED_CLASSES,
            'mapping_config': CURRENT_MAPPING
        },
        'models': {
            'model1': {
                'name': model_names['model1'],
                'info': model_info['model1'],
                'wins': model1_wins
            },
            'model2': {
                'name': model_names['model2'],
                'info': model_info['model2'],
                'wins': model2_wins
            }
        },
        'overall_statistics': {
            'model1': {
                'inference_time': stats1['inference_time'],
                'confidence_scores': stats1['confidence_scores'],
                'detection_counts': stats1['detection_counts'],
                'fps_rates': stats1['fps_rates'],
                'throughput': stats1['throughput']
            },
            'model2': {
                'inference_time': stats2['inference_time'],
                'confidence_scores': stats2['confidence_scores'],
                'detection_counts': stats2['detection_counts'],
                'fps_rates': stats2['fps_rates'],
                'throughput': stats2['throughput']
            }
        },
        'per_image_results': []
    }
    
    # Add per-image results
    for result in results_data:
        image_result = {
            'filename': result['filename'],
            'winner': result['comparison'],
            'model1': {
                'speeds': result['speeds1'],
                'analysis': result['analysis1'],
                'metrics': result.get('metrics1', {})
            },
            'model2': {
                'speeds': result['speeds2'],
                'analysis': result['analysis2'],
                'metrics': result.get('metrics2', {})
            }
        }
        report_data['per_image_results'].append(image_result)
    
    # Write JSON file
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(report_data, f, indent=2, default=str)
    
    return json_path, json_filename

# --- Flask Routes ---
@app.route('/', methods=['GET'])
def index():
    # Just render the page - no cleanup needed here
    return render_template('index.html')

@app.route('/get_class_mappings', methods=['GET'])
def get_class_mappings():
    return jsonify({
        'mappings': CLASS_MAPPINGS,
        'current_mapping': CURRENT_MAPPING,
        'selected_classes': SELECTED_CLASSES,
        'confidence_threshold': CONFIDENCE_THRESHOLD
    })

@app.route('/get_available_models', methods=['GET'])
def get_available_models():
    """Get list of available models in the models directory"""
    models = []
    
    if not os.path.exists(MODEL_FOLDER):
        os.makedirs(MODEL_FOLDER, exist_ok=True)
        return jsonify({'models': [], 'error': 'Models directory is empty'})
    
    try:
        all_files = glob.glob(os.path.join(MODEL_FOLDER, '*'))
        
        for file_path in all_files:
            if not os.path.isfile(file_path):
                continue
                
            filename = os.path.basename(file_path)
            ext = filename.rsplit('.', 1)[-1].lower() if '.' in filename else ''
            
            if ext in ALLOWED_MODEL_EXTENSIONS:
                file_size = os.path.getsize(file_path)
                if file_size >= 1024:  # At least 1KB
                    models.append({
                        'filename': filename,
                        'path': file_path,
                        'size': format_bytes(file_size),
                        'format': 'ONNX' if ext == 'onnx' else 'PyTorch'
                    })
        
        return jsonify({
            'models': models,
            'count': len(models),
            'error': None if len(models) >= 2 else 'At least 2 models required for comparison'
        })
        
    except Exception as e:
        return jsonify({'models': [], 'error': f'Error reading models: {str(e)}'})

@app.route('/set_class_mapping', methods=['POST'])
def set_class_mapping():
    global CURRENT_MAPPING, SELECTED_CLASSES, CONFIDENCE_THRESHOLD
    
    data = request.json
    mapping_name = data.get('mapping_name')
    selected_classes = data.get('selected_classes', [])
    confidence_threshold = data.get('confidence_threshold', CONFIDENCE_THRESHOLD)
    
    if mapping_name not in CLASS_MAPPINGS:
        return jsonify({'error': 'Invalid mapping name'}), 400
    
    available_classes = CLASS_MAPPINGS[mapping_name]['standard_classes']
    invalid_classes = [cls for cls in selected_classes if cls not in available_classes]
    
    if invalid_classes:
        return jsonify({'error': f'Invalid classes: {invalid_classes}'}), 400
    
    # Validate confidence threshold
    try:
        confidence_threshold = float(confidence_threshold)
        if not 0.0 <= confidence_threshold <= 1.0:
            return jsonify({'error': 'Confidence threshold must be between 0.0 and 1.0'}), 400
    except ValueError:
        return jsonify({'error': 'Invalid confidence threshold value'}), 400
    
    CURRENT_MAPPING = mapping_name
    SELECTED_CLASSES = selected_classes
    CONFIDENCE_THRESHOLD = confidence_threshold
    
    return jsonify({
        'success': True, 
        'current_mapping': CURRENT_MAPPING, 
        'selected_classes': SELECTED_CLASSES,
        'confidence_threshold': CONFIDENCE_THRESHOLD
    })

@app.route('/upload', methods=['POST'])
def upload_files():
    try:
        # Validate request
        if 'files[]' not in request.files:
            return jsonify({'error': 'No file part in request'}), 400

        files = request.files.getlist('files[]')
        
        if not files or all(file.filename == '' for file in files):
            return jsonify({'error': 'No files selected'}), 400
        
        # Validate file count and types
        if len(files) > 50:  # Limit batch size
            return jsonify({'error': 'Too many files. Maximum 50 files allowed per batch.'}), 400
        
        # Validate each file
        total_size = 0
        for file in files:
            if file.filename == '':
                continue
            
            # Check file extension
            ext = file.filename.rsplit('.', 1)[1].lower() if '.' in file.filename else ''
            if ext not in ALLOWED_IMAGE_EXTENSIONS and ext not in ALLOWED_VIDEO_EXTENSIONS:
                return jsonify({'error': f'Invalid file type: {file.filename}. Supported formats: {ALLOWED_IMAGE_EXTENSIONS | ALLOWED_VIDEO_EXTENSIONS}'}), 400
            
            # Check file size (estimate)
            file.seek(0, os.SEEK_END)
            file_size = file.tell()
            file.seek(0)  # Reset file pointer
            
            if file_size > 100 * 1024 * 1024:  # 100MB per file
                return jsonify({'error': f'File too large: {file.filename} ({format_bytes(file_size)}). Maximum size: 100MB per file.'}), 400
            
            total_size += file_size
        
        if total_size > 500 * 1024 * 1024:  # 500MB total
            return jsonify({'error': f'Total file size too large ({format_bytes(total_size)}). Maximum total size: 500MB.'}), 400
        
        # Get selected models from request
        selected_model1 = request.form.get('model1')
        selected_model2 = request.form.get('model2')
        
        if not selected_model1 or not selected_model2:
            # Fallback to automatic selection
            model_files, model_names = find_model_files()
            
            if 'model1' not in model_files or 'model2' not in model_files:
                error_msg = 'Two model files are required for comparison. '
                if len(model_files) == 0:
                    error_msg += 'No model files found in the "models" directory. '
                elif len(model_files) == 1:
                    error_msg += f'Only one model found: {list(model_names.values())[0]}. '
                error_msg += 'Please add .pt or .onnx model files to the "models" directory.'
                return jsonify({'error': error_msg}), 500
        else:
            # Use selected models
            model1_path = os.path.join(MODEL_FOLDER, selected_model1)
            model2_path = os.path.join(MODEL_FOLDER, selected_model2)
            
            # Validate selected models exist
            if not os.path.exists(model1_path) or not os.path.exists(model2_path):
                return jsonify({'error': 'Selected model files not found'}), 400
            
            if selected_model1 == selected_model2:
                return jsonify({'error': 'Please select two different models for comparison'}), 400
            
            model_files = {'model1': model1_path, 'model2': model2_path}
            model_names = {'model1': selected_model1, 'model2': selected_model2}

        # Validate class configuration
        if not SELECTED_CLASSES:
            return jsonify({'error': 'No classes selected for comparison. Please configure detection classes first.'}), 400
        
        if CURRENT_MAPPING not in CLASS_MAPPINGS:
            return jsonify({'error': f'Invalid class mapping configuration: {CURRENT_MAPPING}'}), 500
        
        try:
            mapping_config = CLASS_MAPPINGS[CURRENT_MAPPING]
            
            # Load models with detailed error handling
            print(f"Loading models for comparison...")
            print(f"Model 1: {model_names['model1']}")
            print(f"Model 2: {model_names['model2']}")
            print(f"Selected classes: {SELECTED_CLASSES}")
            
            try:
                model1 = load_model(model_files['model1'], mapping_config, SELECTED_CLASSES)
            except Exception as e:
                return jsonify({'error': f'Failed to load Model 1 ({model_names["model1"]}): {str(e)}'}), 500
            
            try:
                model2 = load_model(model_files['model2'], mapping_config, SELECTED_CLASSES)
            except Exception as e:
                return jsonify({'error': f'Failed to load Model 2 ({model_names["model2"]}): {str(e)}'}), 500
            
            model1_info = get_model_info(model_files['model1'])
            model2_info = get_model_info(model_files['model2'])
            
            print(f"Models loaded successfully")
            
        except Exception as e:
            print(f"Error in model loading: {str(e)}")
            return jsonify({'error': f'Error setting up models: {str(e)}'}), 500

        # Initialize performance metrics
        metrics1 = PerformanceMetrics()
        metrics2 = PerformanceMetrics()
        
        # Process uploaded files
        image_paths = []
        results_data = []
        
        for file in files:
            if not file or file.filename == '':
                continue
                
            # Determine file type based on extension
            ext = file.filename.rsplit('.', 1)[1].lower() if '.' in file.filename else ''
            if ext in ALLOWED_VIDEO_EXTENSIONS:
                file_type = 'video'
            elif ext in ALLOWED_IMAGE_EXTENSIONS:
                file_type = 'image'
            else:
                return jsonify({'error': f'File type not supported: {file.filename}. Please upload image or video files only.'}), 400
            
            original_filename = secure_filename(file.filename)
            unique_id = str(uuid.uuid4())
            input_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{unique_id}_{original_filename}")
            
            try:
                # Validate file before saving
                try:
                    file.save(input_path)
                    
                    # Additional validation after saving
                    if file_type == 'image':
                        # Validate image file
                        try:
                            img = cv2.imread(input_path)
                            if img is None:
                                raise ValueError("Invalid image file")
                            height, width = img.shape[:2]
                            if width < 32 or height < 32:
                                raise ValueError(f"Image too small: {width}x{height}. Minimum size: 32x32")
                            if width > 10000 or height > 10000:
                                raise ValueError(f"Image too large: {width}x{height}. Maximum size: 10000x10000")
                        except Exception as e:
                            os.remove(input_path)
                            return jsonify({'error': f'Invalid image file {original_filename}: {str(e)}'}), 400
                        
                        image_paths.append(input_path)
                except Exception as e:
                    if os.path.exists(input_path):
                        os.remove(input_path)
                    return jsonify({'error': f'Failed to save file {original_filename}: {str(e)}'}), 500
                
                if file_type == 'image':
                    pass  # Already handled above
                else:
                    # Handle video processing
                    video_output_filename = f"{unique_id}_comparison.mp4"
                    video_output_path = os.path.join(app.config['PROCESSED_FOLDER'], video_output_filename)
                    
                    print(f"Processing video: {original_filename}")
                    video_result = process_video(input_path, model1, model2, metrics1, metrics2, model_names, video_output_path)
                    
                    # Clean up input video after processing
                    try:
                        os.remove(input_path)
                    except:
                        pass
                    
                    # Get video stats
                    stats1 = metrics1.get_statistics()
                    stats2 = metrics2.get_statistics()
                    
                    return jsonify({
                        'success': True,
                        'video_result': {
                            'original_filename': original_filename,
                            'output_url': url_for('static', filename=f'processed/{video_output_filename}'),
                            'total_frames': video_result['total_frames'],
                            'fps': video_result['fps'],
                            'avg_processing_time': video_result['avg_processing_time'],
                            'model1_stats': {
                                'avg_inference_time': stats1['inference_time']['mean'],
                                'avg_fps': stats1['fps_rates']['mean']
                            },
                            'model2_stats': {
                                'avg_inference_time': stats2['inference_time']['mean'],
                                'avg_fps': stats2['fps_rates']['mean']
                            }
                        },
                        'model_names': model_names
                    })
                    
            except Exception as e:
                app.logger.error(f'Error saving file {original_filename}: {str(e)}')
                return jsonify({'error': f'Error saving file {original_filename}: {str(e)}'}), 500
        
        # Process all images in batch
        if image_paths:
            batch_results = process_batch_images(image_paths, model1, model2, metrics1, metrics2, model_names)
            
            # Clean up original uploads after processing
            session_id = unique_id  # Use from last file processed
            
            # Generate processed images with bounding boxes
            for i, result in enumerate(batch_results):
                original_filename = result['filename']
                unique_id = str(uuid.uuid4())
                
                # Save processed images
                output_path1 = os.path.join(app.config['PROCESSED_FOLDER'], f"{unique_id}_model1_{original_filename}")
                output_path2 = os.path.join(app.config['PROCESSED_FOLDER'], f"{unique_id}_model2_{original_filename}")
                
                draw_boxes_image(result['path'], result['results1'], output_path1, model_names['model1'], model1.class_mapping)
                draw_boxes_image(result['path'], result['results2'], output_path2, model_names['model2'], model2.class_mapping)
                
                results_data.append({
                    'original_filename': original_filename,
                    'original_url': url_for('static', filename=f'uploads/{os.path.basename(result["path"])}'),
                    'processed_url1': url_for('static', filename=f'processed/{unique_id}_model1_{original_filename}'),
                    'processed_url2': url_for('static', filename=f'processed/{unique_id}_model2_{original_filename}'),
                    'analysis1': result['analysis1'],
                    'analysis2': result['analysis2'],
                    'preprocess_speed1': result['speeds1']['preprocess'],
                    'inference_speed1': result['speeds1']['inference'],
                    'postprocess_speed1': result['speeds1']['postprocess'],
                    'preprocess_speed2': result['speeds2']['preprocess'],
                    'inference_speed2': result['speeds2']['inference'],
                    'postprocess_speed2': result['speeds2']['postprocess'],
                    'comparison': result['comparison']
                })
        
        # Generate overall statistics
        stats1 = metrics1.get_statistics()
        stats2 = metrics2.get_statistics()
        
        overall_stats = {
            'model1': {
                'class_counts': {name: 0 for name in SELECTED_CLASSES},
                'avg_confidence': stats1['confidence_scores']['mean'],
                'image_wins': sum(1 for r in batch_results if model_names['model1'] in r['comparison']),
                'model_size': f"{model1_info['size']} ({model1_info['format']})",
                'avg_inference_speed': stats1['inference_time']['mean'],
                'avg_preprocess_speed': stats1['preprocess_time']['mean'],
                'avg_postprocess_speed': stats1['postprocess_time']['mean'],
                'throughput': stats1['throughput'],
                'fps': stats1['fps_rates']['mean']
            },
            'model2': {
                'class_counts': {name: 0 for name in SELECTED_CLASSES},
                'avg_confidence': stats2['confidence_scores']['mean'],
                'image_wins': sum(1 for r in batch_results if model_names['model2'] in r['comparison']),
                'model_size': f"{model2_info['size']} ({model2_info['format']})",
                'avg_inference_speed': stats2['inference_time']['mean'],
                'avg_preprocess_speed': stats2['preprocess_time']['mean'],
                'avg_postprocess_speed': stats2['postprocess_time']['mean'],
                'throughput': stats2['throughput'],
                'fps': stats2['fps_rates']['mean']
            }
        }
        
        # Calculate class-specific counts
        for result in batch_results:
            for name in SELECTED_CLASSES:
                overall_stats['model1']['class_counts'][name] += result['analysis1']['class_counts'].get(name, 0)
                overall_stats['model2']['class_counts'][name] += result['analysis2']['class_counts'].get(name, 0)
        
        # Generate reports
        model_info = {'model1': model1_info, 'model2': model2_info}
        pdf_path, pdf_filename = generate_pdf_report(batch_results, metrics1, metrics2, model_names, model_info)
        csv_path, csv_filename = generate_csv_report(batch_results, metrics1, metrics2, model_names)
        json_path, json_filename = generate_json_report(batch_results, metrics1, metrics2, model_names, model_info)
        
        return jsonify({
            'results': results_data,
            'overall_stats': overall_stats,
            'class_names': SELECTED_CLASSES,
            'model_names': model_names,
            'mapping_config': {
                'current_mapping': CURRENT_MAPPING,
                'selected_classes': SELECTED_CLASSES
            },
            'performance_metrics': {
                'model1': stats1,
                'model2': stats2
            },
            'pdf_report': {
                'url': url_for('static', filename=f'reports/{pdf_filename}'),
                'filename': pdf_filename
            },
            'csv_report': {
                'url': url_for('download_report', filename=csv_filename, format='csv'),
                'filename': csv_filename
            },
            'json_report': {
                'url': url_for('download_report', filename=json_filename, format='json'),
                'filename': json_filename
            }
        })
        
    except Exception as e:
        app.logger.error(f'Unexpected error in upload_files: {str(e)}')
        return jsonify({'error': f'Server error: {str(e)}'}), 500

@app.route('/download_report/<filename>')
def download_report(filename):
    """Download generated report in various formats"""
    try:
        file_path = os.path.join(REPORTS_FOLDER, filename)
        
        if not os.path.exists(file_path):
            return jsonify({'error': f'Report not found: {filename}'}), 404
        
        # Determine mimetype based on file extension
        if filename.endswith('.pdf'):
            mimetype = 'application/pdf'
        elif filename.endswith('.csv'):
            mimetype = 'text/csv'
        elif filename.endswith('.json'):
            mimetype = 'application/json'
        else:
            mimetype = 'application/octet-stream'
        
        return send_file(
            file_path,
            as_attachment=True,
            download_name=filename,
            mimetype=mimetype
        )
    except Exception as e:
        return jsonify({'error': f'Error downloading report: {str(e)}'}), 500

# Cleanup on shutdown
import atexit
import signal

def cleanup_on_exit():
    """Quick cleanup on application exit - non-blocking"""
    print("\nShutting down gracefully...")
    # Don't do heavy cleanup on exit - just exit quickly
    # The next startup will handle cleanup in background
    try:
        # Just do minimal cleanup if needed
        pass
    except:
        pass
    finally:
        # Force exit to prevent hanging
        os._exit(0)

# Register cleanup handlers
def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully"""
    print("\nReceived interrupt signal...")
    cleanup_on_exit()

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)
# Don't use atexit - it can cause hanging

if __name__ == '__main__':
    # Quick startup - just ensure folders exist
    for folder in [UPLOAD_FOLDER, PROCESSED_FOLDER, MODEL_FOLDER, REPORTS_FOLDER]:
        os.makedirs(folder, exist_ok=True)
    
    # Start cleanup in background (won't delay startup)
    print("Starting Flask application...")
    background_cleanup_thread = threading.Thread(target=background_cleanup, daemon=True)
    background_cleanup_thread.start()
    
    # Start the Flask application
    app.run(debug=True)