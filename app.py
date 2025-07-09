import os
import time
from flask import Flask, request, jsonify, render_template, url_for
from werkzeug.utils import secure_filename
from ultralytics import YOLO
import cv2
import numpy as np
import uuid
import shutil

# --- Configuration ---
UPLOAD_FOLDER = 'static/uploads'
PROCESSED_FOLDER = 'static/processed'
MODEL_FOLDER = 'models'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# --- App Initialization ---
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER
app.config['MODEL_FOLDER'] = MODEL_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max upload size

# --- Helper Functions ---
def allowed_file(filename):
    """Checks if the file has an allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def clear_folders():
    """Clears the upload and processed directories for a fresh start."""
    for folder in [UPLOAD_FOLDER, PROCESSED_FOLDER]:
        if os.path.exists(folder):
            shutil.rmtree(folder)
        os.makedirs(folder, exist_ok=True)

def format_bytes(size):
    """Formats file size in bytes to a human-readable string (KB, MB, GB)."""
    if size == 0:
        return "0B"
    power = 1024
    n = 0
    power_labels = {0: 'B', 1: 'KB', 2: 'MB', 3: 'GB', 4: 'TB'}
    while size >= power and n < len(power_labels) -1 :
        size /= power
        n += 1
    return f"{size:.2f} {power_labels[n]}"

def draw_boxes(image_path, results, output_path, model_name):
    """Draws bounding boxes on an image and saves it."""
    image = cv2.imread(image_path)
    class_names = results[0].names if isinstance(results[0].names, dict) else {}
    
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        confidence = box.conf[0]
        cls_id = int(box.cls[0])
        label = f"{class_names.get(cls_id, 'unknown')} {confidence:.2f}"
        
        # Color coding for models: Green for Model 1, Blue for Model 2
        color = (0, 255, 0) if "1" in model_name else (0, 0, 255)
        
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
    cv2.imwrite(output_path, image)

def analyze_results(results, class_names):
    """
    Analyzes detection results to extract counts and confidences.
    Normalizes class names to lowercase for consistency.
    """
    lower_class_names = {k: v.lower() for k, v in class_names.items()}
    all_names = sorted(list(set(lower_class_names.values())))

    analysis = {
        'total_detections': len(results[0].boxes),
        'class_counts': {name: 0 for name in all_names},
        'avg_confidence': 0,
        'avg_class_confidence': {name: 0 for name in all_names},
        'confidences': []
    }
    
    if analysis['total_detections'] == 0:
        return analysis

    total_conf = 0
    class_conf_totals = {name: 0 for name in all_names}
    
    for box in results[0].boxes:
        confidence = box.conf[0].item()
        cls_id = int(box.cls[0])
        cls_name = lower_class_names.get(cls_id, 'unknown')

        analysis['confidences'].append(confidence)
        total_conf += confidence
        if cls_name != 'unknown':
            analysis['class_counts'][cls_name] += 1
            class_conf_totals[cls_name] += confidence

    analysis['avg_confidence'] = total_conf / analysis['total_detections']
    
    for name, count in analysis['class_counts'].items():
        if count > 0:
            analysis['avg_class_confidence'][name] = class_conf_totals[name] / count
            
    return analysis

def compare_models(analysis1, analysis2, speed1, speed2):
    """Compares the analysis of two models to determine a 'winner'."""
    score1, score2 = 0, 0

    # Compare total detections
    if analysis1['total_detections'] > analysis2['total_detections']:
        score1 += 1
    elif analysis2['total_detections'] > analysis1['total_detections']:
        score2 += 1

    # Compare average confidence
    if analysis1['avg_confidence'] > analysis2['avg_confidence']:
        score1 += 1
    elif analysis2['avg_confidence'] > analysis1['avg_confidence']:
        score2 += 1
        
    # Compare inference speed (lower is better)
    if speed1 < speed2:
        score1 += 1
    elif speed2 < speed1:
        score2 += 1

    # Compare football detections (using lowercase)
    if analysis1['class_counts'].get('football', 0) > analysis2['class_counts'].get('football', 0):
        score1 += 0.5
    elif analysis2['class_counts'].get('football', 0) > analysis1['class_counts'].get('football', 0):
        score2 += 0.5
        
    # Compare cone detections (using lowercase)
    if analysis1['class_counts'].get('cone', 0) > analysis2['class_counts'].get('cone', 0):
        score1 += 0.5
    elif analysis2['class_counts'].get('cone', 0) > analysis1['class_counts'].get('cone', 0):
        score2 += 0.5

    if score1 > score2:
        return "Model 1 is better"
    if score2 > score1:
        return "Model 2 is better"
    if score1 == score2 and score1 > 0:
        return "Both models performed similarly"
    return "No conclusive winner (or no detections)"


# --- Flask Routes ---
@app.route('/', methods=['GET'])
def index():
    """Renders the main page."""
    clear_folders()
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_files():
    """Handles file uploads and triggers model inference and comparison."""
    if 'files[]' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    files = request.files.getlist('files[]')
    
    # --- Load Models & Get Size ---
    model1_path = os.path.join(app.config['MODEL_FOLDER'], 'best_model_1.pt')
    model2_path = os.path.join(app.config['MODEL_FOLDER'], 'best_model_2.pt')

    if not os.path.exists(model1_path) or not os.path.exists(model2_path):
        return jsonify({'error': 'Model files not found. Please place best_model_1.pt and best_model_2.pt in the "models" directory.'}), 500

    model1_size = format_bytes(os.path.getsize(model1_path))
    model2_size = format_bytes(os.path.getsize(model2_path))

    try:
        model1 = YOLO(model1_path)
        model2 = YOLO(model2_path)
        class_names = model1.names 
        class_name_list = sorted(list(set(v.lower() for v in class_names.values())))
    except Exception as e:
        return jsonify({'error': f'Error loading models: {str(e)}'}), 500

    results_data = []
    # Add accumulators for detailed speed stats
    overall_stats = {
        'model1': {'total_detections': 0, 'class_counts': {name: 0 for name in class_name_list}, 'total_confidence': 0, 'image_wins': 0, 'model_size': model1_size, 'total_preprocess_speed': 0, 'total_inference_speed': 0, 'total_postprocess_speed': 0},
        'model2': {'total_detections': 0, 'class_counts': {name: 0 for name in class_name_list}, 'total_confidence': 0, 'image_wins': 0, 'model_size': model2_size, 'total_preprocess_speed': 0, 'total_inference_speed': 0, 'total_postprocess_speed': 0}
    }

    for file in files:
        if file and allowed_file(file.filename):
            original_filename = secure_filename(file.filename)
            unique_id = str(uuid.uuid4())
            input_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{unique_id}_{original_filename}")
            file.save(input_path)

            try:
                # Get detailed speed breakdown for model 1
                results1 = model1(input_path, verbose=False)
                speed1_dict = results1[0].speed
                preprocess_speed1 = speed1_dict.get('preprocess', 0)
                inference_speed1 = speed1_dict.get('inference', 0)
                postprocess_speed1 = speed1_dict.get('postprocess', 0)
                
                # Get detailed speed breakdown for model 2
                results2 = model2(input_path, verbose=False)
                speed2_dict = results2[0].speed
                preprocess_speed2 = speed2_dict.get('preprocess', 0)
                inference_speed2 = speed2_dict.get('inference', 0)
                postprocess_speed2 = speed2_dict.get('postprocess', 0)

            except Exception as e:
                return jsonify({'error': f'Error during model inference: {str(e)}'}), 500

            # --- Analyze and Compare ---
            analysis1 = analyze_results(results1, class_names)
            analysis2 = analyze_results(results2, class_names)
            # Compare based on main inference speed
            comparison = compare_models(analysis1, analysis2, inference_speed1, inference_speed2)

            # --- Update Overall Stats ---
            overall_stats['model1']['total_detections'] += analysis1['total_detections']
            overall_stats['model2']['total_detections'] += analysis2['total_detections']
            overall_stats['model1']['total_confidence'] += sum(analysis1['confidences'])
            overall_stats['model2']['total_confidence'] += sum(analysis2['confidences'])
            
            # Accumulate detailed speeds
            overall_stats['model1']['total_preprocess_speed'] += preprocess_speed1
            overall_stats['model1']['total_inference_speed'] += inference_speed1
            overall_stats['model1']['total_postprocess_speed'] += postprocess_speed1
            overall_stats['model2']['total_preprocess_speed'] += preprocess_speed2
            overall_stats['model2']['total_inference_speed'] += inference_speed2
            overall_stats['model2']['total_postprocess_speed'] += postprocess_speed2

            if "Model 1" in comparison:
                overall_stats['model1']['image_wins'] += 1
            elif "Model 2" in comparison:
                overall_stats['model2']['image_wins'] += 1
            
            for name in class_name_list:
                overall_stats['model1']['class_counts'][name] += analysis1['class_counts'].get(name, 0)
                overall_stats['model2']['class_counts'][name] += analysis2['class_counts'].get(name, 0)

            # --- Save Processed Images ---
            output_path1 = os.path.join(app.config['PROCESSED_FOLDER'], f"{unique_id}_model1_{original_filename}")
            output_path2 = os.path.join(app.config['PROCESSED_FOLDER'], f"{unique_id}_model2_{original_filename}")
            draw_boxes(input_path, results1, output_path1, "Model 1")
            draw_boxes(input_path, results2, output_path2, "Model 2")

            results_data.append({
                'original_filename': original_filename,
                'original_url': url_for('static', filename=f'uploads/{unique_id}_{original_filename}'),
                'processed_url1': url_for('static', filename=f'processed/{unique_id}_model1_{original_filename}'),
                'processed_url2': url_for('static', filename=f'processed/{unique_id}_model2_{original_filename}'),
                'analysis1': analysis1,
                'analysis2': analysis2,
                'preprocess_speed1': preprocess_speed1,
                'inference_speed1': inference_speed1,
                'postprocess_speed1': postprocess_speed1,
                'preprocess_speed2': preprocess_speed2,
                'inference_speed2': inference_speed2,
                'postprocess_speed2': postprocess_speed2,
                'comparison': comparison
            })

    # Calculate final overall stats
    num_images = len(results_data)
    if num_images > 0:
        # Model 1 averages
        if overall_stats['model1']['total_detections'] > 0:
            overall_stats['model1']['avg_confidence'] = overall_stats['model1']['total_confidence'] / overall_stats['model1']['total_detections']
        else:
            overall_stats['model1']['avg_confidence'] = 0
        overall_stats['model1']['avg_preprocess_speed'] = overall_stats['model1']['total_preprocess_speed'] / num_images
        overall_stats['model1']['avg_inference_speed'] = overall_stats['model1']['total_inference_speed'] / num_images
        overall_stats['model1']['avg_postprocess_speed'] = overall_stats['model1']['total_postprocess_speed'] / num_images
            
        # Model 2 averages
        if overall_stats['model2']['total_detections'] > 0:
            overall_stats['model2']['avg_confidence'] = overall_stats['model2']['total_confidence'] / overall_stats['model2']['total_detections']
        else:
            overall_stats['model2']['avg_confidence'] = 0
        overall_stats['model2']['avg_preprocess_speed'] = overall_stats['model2']['total_preprocess_speed'] / num_images
        overall_stats['model2']['avg_inference_speed'] = overall_stats['model2']['total_inference_speed'] / num_images
        overall_stats['model2']['avg_postprocess_speed'] = overall_stats['model2']['total_postprocess_speed'] / num_images

    return jsonify({'results': results_data, 'overall_stats': overall_stats, 'class_names': class_name_list})

if __name__ == '__main__':
    for folder in [UPLOAD_FOLDER, PROCESSED_FOLDER, MODEL_FOLDER]:
        os.makedirs(folder, exist_ok=True)
    
    if not os.path.exists(os.path.join(MODEL_FOLDER, 'best_model_1.pt')):
        with open(os.path.join(MODEL_FOLDER, 'best_model_1.pt'), 'w') as f:
            f.write("This is a placeholder. Replace with your actual YOLO model file.")
    if not os.path.exists(os.path.join(MODEL_FOLDER, 'best_model_2.pt')):
        with open(os.path.join(MODEL_FOLDER, 'best_model_2.pt'), 'w') as f:
            f.write("This is a placeholder. Replace with your actual YOLO model file.")

    app.run(debug=True)
