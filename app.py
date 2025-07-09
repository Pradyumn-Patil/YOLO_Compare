import os
import time
import json
import statistics
from datetime import datetime
from flask import Flask, request, jsonify, render_template, url_for, send_file
from werkzeug.utils import secure_filename
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
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
from PIL import Image
import io
import base64

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
    if '.' not in filename:
        return False
    ext = filename.rsplit('.', 1)[1].lower()
    if file_type == 'image':
        return ext in ALLOWED_IMAGE_EXTENSIONS
    elif file_type == 'video':
        return ext in ALLOWED_VIDEO_EXTENSIONS
    return False

def clear_folders():
    for folder in [UPLOAD_FOLDER, PROCESSED_FOLDER, REPORTS_FOLDER]:
        if os.path.exists(folder):
            shutil.rmtree(folder)
        os.makedirs(folder, exist_ok=True)

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
    if not hasattr(model, 'names') or not model.names:
        if hasattr(model, 'model_path'):
            filename = os.path.basename(model.model_path).lower()
        else:
            filename = 'unknown'
            
        if 'trtfootballyolo' in filename or 'football' in filename:
            default_mapping = {0: 'cone', 1: 'football'}
            model.names = default_mapping
            print(f"Set default class names for football model: {model.names}")
        else:
            default_mapping = {i: f'class_{i}' for i in range(len(selected_classes))}
            model.names = default_mapping
            print(f"Set generic class names: {model.names}")
    
    standardized_mapping = {}
    
    for class_id, class_name in model.names.items():
        normalized_name = normalize_class_name(class_name, mapping_config)
        
        if normalized_name in selected_classes:
            standardized_mapping[class_id] = normalized_name
        else:
            if class_name.lower() in selected_classes:
                standardized_mapping[class_id] = class_name.lower()
    
    if not standardized_mapping:
        print(f"No class mappings found, attempting direct mapping...")
        print(f"Model classes: {model.names}")
        print(f"Selected classes: {selected_classes}")
        
        if len(model.names) == len(selected_classes):
            for i, selected_class in enumerate(selected_classes):
                if i in model.names:
                    standardized_mapping[i] = selected_class
                    print(f"Direct mapping: {i} -> {selected_class}")
    
    print(f"Final standardized mapping: {standardized_mapping}")
    return standardized_mapping

def detect_model_type(model_path):
    filename = os.path.basename(model_path).lower()
    
    if 'trtfootballyolo' in filename or 'football' in filename:
        return 'yolov8_football'
    elif 'cone' in filename:
        return 'yolov8_football'
    elif 'coco' in filename:
        return 'coco'
    else:
        return 'custom_football'

def find_model_files():
    model_files = {}
    model_names = {}
    
    all_files = glob.glob(os.path.join(MODEL_FOLDER, '*'))
    
    supported_models = [
        f for f in all_files 
        if f.rsplit('.', 1)[-1].lower() in ALLOWED_MODEL_EXTENSIONS
    ]
    
    if len(supported_models) >= 2:
        supported_models.sort()
        
        model_files['model1'] = supported_models[0]
        model_names['model1'] = os.path.basename(supported_models[0])
        
        model_files['model2'] = supported_models[1]
        model_names['model2'] = os.path.basename(supported_models[1])
        
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
    if os.path.getsize(model_path) < 1024:
        raise ValueError(f"Model file '{os.path.basename(model_path)}' is too small. Please replace the placeholder with a real model file.")

    try:
        model = YOLO(model_path, task='detect')
        
        if model_path.endswith('.onnx'):
            print(f"Loading ONNX model: {os.path.basename(model_path)}")
            
            try:
                import tempfile
                import numpy as np
                
                dummy_image = np.zeros((640, 640, 3), dtype=np.uint8)
                temp_path = tempfile.mktemp(suffix='.jpg')
                cv2.imwrite(temp_path, dummy_image)
                
                try:
                    results = model(temp_path, verbose=False)
                    if results and len(results) > 0:
                        result = results[0]
                        if hasattr(result, 'names'):
                            print(f"Model classes detected: {result.names}")
                        else:
                            print("Model doesn't have class names, will use default mapping")
                except Exception as e:
                    print(f"Test inference failed: {e}")
                
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                    
            except Exception as e:
                print(f"Error testing model structure: {e}")
        
        model_type = detect_model_type(model_path)
        print(f"Detected model type: {model_type} for {os.path.basename(model_path)}")
        
        class_mapping = map_model_classes(model, mapping_config, selected_classes)
        
        model.class_mapping = class_mapping
        model.selected_classes = selected_classes
        
        print(f"Applied class mapping: {class_mapping}")
        
        return model
        
    except Exception as e:
        print(f"Error loading model {model_path}: {e}")
        try:
            model = YOLO(model_path)
            
            if 'trtfootballyolo' in model_path.lower():
                model.names = {0: 'cone', 1: 'football'}
                print(f"Manually set class names for {os.path.basename(model_path)}: {model.names}")
            
            class_mapping = map_model_classes(model, mapping_config, selected_classes)
            model.class_mapping = class_mapping
            model.selected_classes = selected_classes
            
            print(f"Applied class mapping after manual setup: {class_mapping}")
            return model
            
        except Exception as e2:
            raise ValueError(f"Failed to load model {os.path.basename(model_path)}: {e2}")
    
    return model

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

def process_batch_images(image_paths, model1, model2, metrics1, metrics2):
    """Process a batch of images and collect detailed metrics"""
    results_data = []
    
    for image_path in image_paths:
        filename = os.path.basename(image_path)
        
        # Process with model 1
        start_time = time.time()
        try:
            results1 = model1(image_path, verbose=False)
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
            
            # Update metrics
            metrics1.add_inference_time(inference_time1)
            metrics1.add_preprocess_time(preprocess_speed1)
            metrics1.add_postprocess_time(postprocess_speed1)
            metrics1.add_total_time(total_speed1)
            metrics1.add_confidence_scores(confidences1)
            metrics1.add_detection_count(detection_count1)
            if total_speed1 > 0:
                metrics1.add_fps_rate(1000 / total_speed1)
            
        except Exception as e:
            print(f"Error processing {filename} with model 1: {e}")
            filtered_results1 = []
            inference_time1 = preprocess_speed1 = postprocess_speed1 = 0
            confidences1 = []
            detection_count1 = 0
        
        # Process with model 2
        start_time = time.time()
        try:
            results2 = model2(image_path, verbose=False)
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
            
            # Update metrics
            metrics2.add_inference_time(inference_time2)
            metrics2.add_preprocess_time(preprocess_speed2)
            metrics2.add_postprocess_time(postprocess_speed2)
            metrics2.add_total_time(total_speed2)
            metrics2.add_confidence_scores(confidences2)
            metrics2.add_detection_count(detection_count2)
            if total_speed2 > 0:
                metrics2.add_fps_rate(1000 / total_speed2)
            
        except Exception as e:
            print(f"Error processing {filename} with model 2: {e}")
            filtered_results2 = []
            inference_time2 = preprocess_speed2 = postprocess_speed2 = 0
            confidences2 = []
            detection_count2 = 0
        
        # Analyze results
        analysis1 = analyze_results(filtered_results1, model1.class_mapping, SELECTED_CLASSES)
        analysis2 = analyze_results(filtered_results2, model2.class_mapping, SELECTED_CLASSES)
        comparison = compare_models(analysis1, analysis2, inference_time1, inference_time2, SELECTED_CLASSES)
        
        results_data.append({
            'filename': filename,
            'path': image_path,
            'results1': filtered_results1,
            'results2': filtered_results2,
            'analysis1': analysis1,
            'analysis2': analysis2,
            'speeds1': {
                'preprocess': preprocess_speed1,
                'inference': inference_time1,
                'postprocess': postprocess_speed1,
                'total': total_speed1
            },
            'speeds2': {
                'preprocess': preprocess_speed2,
                'inference': inference_time2,
                'postprocess': postprocess_speed2,
                'total': total_speed2
            },
            'comparison': comparison
        })
    
    return results_data

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

def compare_models(analysis1, analysis2, speed1, speed2, selected_classes):
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
        return "Model 1 is better"
    if score2 > score1:
        return "Model 2 is better"
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
    model1_wins = sum(1 for r in results_data if "Model 1" in r['comparison'])
    model2_wins = sum(1 for r in results_data if "Model 2" in r['comparison'])
    
    if model1_wins > model2_wins:
        winner = f"Model 1 ({model_names['model1']})"
    elif model2_wins > model1_wins:
        winner = f"Model 2 ({model_names['model2']})"
    else:
        winner = "Tie"
    
    summary_text = f"""
    <b>Overall Winner:</b> {winner}<br/>
    <b>Total Images Processed:</b> {len(results_data)}<br/>
    <b>Model 1 Wins:</b> {model1_wins}<br/>
    <b>Model 2 Wins:</b> {model2_wins}<br/>
    <b>Average Inference Time:</b> Model 1: {stats1['inference_time']['mean']:.2f}ms, Model 2: {stats2['inference_time']['mean']:.2f}ms<br/>
    <b>Average Confidence:</b> Model 1: {stats1['confidence_scores']['mean']:.3f}, Model 2: {stats2['confidence_scores']['mean']:.3f}<br/>
    """
    
    story.append(Paragraph(summary_text, styles['Normal']))
    story.append(Spacer(1, 20))
    
    # Model Information Table
    story.append(Paragraph("Model Information", styles['Heading2']))
    
    model_data = [
        ['Metric', 'Model 1', 'Model 2'],
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
        ['Metric', 'Model 1', 'Model 2', 'Winner'],
        ['Avg Inference Time (ms)', f"{stats1['inference_time']['mean']:.2f} ± {stats1['inference_time']['std']:.2f}", 
         f"{stats2['inference_time']['mean']:.2f} ± {stats2['inference_time']['std']:.2f}",
         'Model 1' if stats1['inference_time']['mean'] < stats2['inference_time']['mean'] else 'Model 2'],
        ['Min Inference Time (ms)', f"{stats1['inference_time']['min']:.2f}", f"{stats2['inference_time']['min']:.2f}",
         'Model 1' if stats1['inference_time']['min'] < stats2['inference_time']['min'] else 'Model 2'],
        ['Max Inference Time (ms)', f"{stats1['inference_time']['max']:.2f}", f"{stats2['inference_time']['max']:.2f}",
         'Model 1' if stats1['inference_time']['max'] < stats2['inference_time']['max'] else 'Model 2'],
        ['Avg Confidence', f"{stats1['confidence_scores']['mean']:.3f} ± {stats1['confidence_scores']['std']:.3f}", 
         f"{stats2['confidence_scores']['mean']:.3f} ± {stats2['confidence_scores']['std']:.3f}",
         'Model 1' if stats1['confidence_scores']['mean'] > stats2['confidence_scores']['mean'] else 'Model 2'],
        ['Avg Detections/Image', f"{stats1['detection_counts']['mean']:.1f} ± {stats1['detection_counts']['std']:.1f}", 
         f"{stats2['detection_counts']['mean']:.1f} ± {stats2['detection_counts']['std']:.1f}",
         'Model 1' if stats1['detection_counts']['mean'] > stats2['detection_counts']['mean'] else 'Model 2'],
        ['Avg FPS', f"{stats1['fps_rates']['mean']:.1f} ± {stats1['fps_rates']['std']:.1f}", 
         f"{stats2['fps_rates']['mean']:.1f} ± {stats2['fps_rates']['std']:.1f}",
         'Model 1' if stats1['fps_rates']['mean'] > stats2['fps_rates']['mean'] else 'Model 2'],
        ['Throughput (images/sec)', f"{stats1['throughput']:.2f}", f"{stats2['throughput']:.2f}",
         'Model 1' if stats1['throughput'] > stats2['throughput'] else 'Model 2']
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
            ['Metric', 'Model 1', 'Model 2'],
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

# --- Flask Routes ---
@app.route('/', methods=['GET'])
def index():
    clear_folders()
    return render_template('index.html')

@app.route('/get_class_mappings', methods=['GET'])
def get_class_mappings():
    return jsonify({
        'mappings': CLASS_MAPPINGS,
        'current_mapping': CURRENT_MAPPING,
        'selected_classes': SELECTED_CLASSES
    })

@app.route('/set_class_mapping', methods=['POST'])
def set_class_mapping():
    global CURRENT_MAPPING, SELECTED_CLASSES
    
    data = request.json
    mapping_name = data.get('mapping_name')
    selected_classes = data.get('selected_classes', [])
    
    if mapping_name not in CLASS_MAPPINGS:
        return jsonify({'error': 'Invalid mapping name'}), 400
    
    available_classes = CLASS_MAPPINGS[mapping_name]['standard_classes']
    invalid_classes = [cls for cls in selected_classes if cls not in available_classes]
    
    if invalid_classes:
        return jsonify({'error': f'Invalid classes: {invalid_classes}'}), 400
    
    CURRENT_MAPPING = mapping_name
    SELECTED_CLASSES = selected_classes
    
    return jsonify({'success': True, 'current_mapping': CURRENT_MAPPING, 'selected_classes': SELECTED_CLASSES})

@app.route('/upload', methods=['POST'])
def upload_files():
    try:
        if 'files[]' not in request.files:
            return jsonify({'error': 'No file part'}), 400

        files = request.files.getlist('files[]')
        media_type = request.form.get('media_type', 'images')
        
        if not files or all(file.filename == '' for file in files):
            return jsonify({'error': 'No files selected'}), 400
        
        # Find and Load Models
        model_files, model_names = find_model_files()
        
        if 'model1' not in model_files or 'model2' not in model_files:
            return jsonify({'error': 'Model files not found. Please place model files in the "models" directory.'}), 500

        try:
            mapping_config = CLASS_MAPPINGS[CURRENT_MAPPING]
            
            model1 = load_model(model_files['model1'], mapping_config, SELECTED_CLASSES)
            model2 = load_model(model_files['model2'], mapping_config, SELECTED_CLASSES)
            
            model1_info = get_model_info(model_files['model1'])
            model2_info = get_model_info(model_files['model2'])
            
        except Exception as e:
            return jsonify({'error': f'Error loading models: {str(e)}'}), 500

        # Initialize performance metrics
        metrics1 = PerformanceMetrics()
        metrics2 = PerformanceMetrics()
        
        # Process uploaded files
        image_paths = []
        results_data = []
        
        for file in files:
            if not file or file.filename == '':
                continue
                
            file_type = 'video' if media_type == 'videos' else 'image'
            
            if not allowed_file(file.filename, file_type):
                return jsonify({'error': f'File type not supported: {file.filename}. Please upload {file_type} files only.'}), 400
            
            original_filename = secure_filename(file.filename)
            unique_id = str(uuid.uuid4())
            input_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{unique_id}_{original_filename}")
            
            try:
                file.save(input_path)
                
                if file_type == 'image':
                    image_paths.append(input_path)
                else:
                    # Handle video processing (simplified for this version)
                    # You can extend this to handle videos similar to the original code
                    pass
                    
            except Exception as e:
                app.logger.error(f'Error saving file {original_filename}: {str(e)}')
                return jsonify({'error': f'Error saving file {original_filename}: {str(e)}'}), 500
        
        # Process all images in batch
        if image_paths:
            batch_results = process_batch_images(image_paths, model1, model2, metrics1, metrics2)
            
            # Generate processed images with bounding boxes
            for i, result in enumerate(batch_results):
                original_filename = result['filename']
                unique_id = str(uuid.uuid4())
                
                # Save processed images
                output_path1 = os.path.join(app.config['PROCESSED_FOLDER'], f"{unique_id}_model1_{original_filename}")
                output_path2 = os.path.join(app.config['PROCESSED_FOLDER'], f"{unique_id}_model2_{original_filename}")
                
                draw_boxes_image(result['path'], result['results1'], output_path1, "Model 1", model1.class_mapping)
                draw_boxes_image(result['path'], result['results2'], output_path2, "Model 2", model2.class_mapping)
                
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
                'image_wins': sum(1 for r in batch_results if "Model 1" in r['comparison']),
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
                'image_wins': sum(1 for r in batch_results if "Model 2" in r['comparison']),
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
        
        # Generate PDF report
        model_info = {'model1': model1_info, 'model2': model2_info}
        pdf_path, pdf_filename = generate_pdf_report(batch_results, metrics1, metrics2, model_names, model_info)
        
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
            }
        })
        
    except Exception as e:
        app.logger.error(f'Unexpected error in upload_files: {str(e)}')
        return jsonify({'error': f'Server error: {str(e)}'}), 500

@app.route('/download_report/<filename>')
def download_report(filename):
    """Download generated PDF report"""
    try:
        return send_file(
            os.path.join(REPORTS_FOLDER, filename),
            as_attachment=True,
            download_name=filename,
            mimetype='application/pdf'
        )
    except Exception as e:
        return jsonify({'error': f'Report not found: {str(e)}'}), 404

if __name__ == '__main__':
    # Ensure all necessary folders exist on startup
    for folder in [UPLOAD_FOLDER, PROCESSED_FOLDER, MODEL_FOLDER, REPORTS_FOLDER]:
        os.makedirs(folder, exist_ok=True)
    
    # Start the Flask application
    app.run(debug=True)