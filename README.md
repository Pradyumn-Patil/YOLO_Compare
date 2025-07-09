# Enhanced YOLO Model Comparison System

## Overview

This enhanced version of the YOLO model comparison system includes sophisticated class mapping capabilities that ensure fair comparison between different models, even when they have different class definitions or naming conventions.

## Key Features

### 🎯 **Class Mapping System**
- **Standardized Classes**: Maps different model classes to standardized names
- **Flexible Aliases**: Handles alternative class names (e.g., "ball" → "football")
- **Selective Comparison**: Choose which specific classes to compare
- **Multiple Domains**: Pre-configured for football, traffic, medical, and general detection

### 🔧 **Configuration Interface**
- **Web-based Setup**: Configure mappings directly in the browser
- **Real-time Updates**: Apply configuration changes without restarting
- **Visual Feedback**: Clear indication of selected classes and mappings

### 📊 **Fair Comparison**
- **Normalized Results**: Only compares selected classes across models
- **Consistent Metrics**: Standardized confidence scores and detection counts
- **Filtered Visualization**: Bounding boxes only show selected classes

## Installation & Setup

### 1. Install Dependencies
```bash
pip install flask ultralytics opencv-python numpy werkzeug
```

### 2. Project Structure
```
project/
├── app.py                 # Main Flask application
├── config.py             # Configuration file (optional)
├── templates/
│   └── index.html        # Web interface
├── static/
│   ├── uploads/          # Uploaded files
│   └── processed/        # Processed results
└── models/               # Place your model files here
    ├── model1.pt         # First model
    └── model2.pt         # Second model
```

### 3. Model Setup
Place your YOLO model files (.pt or .onnx) in the `models/` directory. The system will automatically detect and assign them as Model 1 and Model 2.

## Usage Guide

### 1. Configure Class Mappings

#### Option A: Use Web Interface
1. Start the application: `python app.py`
2. Open http://localhost:5000
3. In the "Model Configuration" section:
   - Select the appropriate detection type (Football, Traffic, etc.)
   - Choose which classes to compare
   - Click "Apply Configuration"

#### Option B: Modify Configuration Code
Edit the `CLASS_MAPPINGS` dictionary in `app.py`:

```python
CLASS_MAPPINGS = {
    'your_custom_detection': {
        'standard_classes': ['class1', 'class2', 'class3'],
        'model_mappings': {
            'model_type_1': {
                0: 'class1',
                1: 'class2',
                2: 'class3'
            }
        },
        'class_aliases': {
            'alternative_name': 'class1',
            'another_name': 'class2'
        }
    }
}
```

### 2. Upload and Compare

1. **Select Media Type**: Choose between Images or Videos
2. **Upload Files**: Select your test files
3. **Run Analysis**: Click "Analyze Performance"
4. **View Results**: Compare model performance across selected classes

## Class Mapping Examples

### Football Detection
```python
'football_detection': {
    'standard_classes': ['football', 'cone', 'player', 'goalpost'],
    'model_mappings': {
        'yolov8_football': {
            0: 'football',
            1: 'cone',
            2: 'player',
            3: 'goalpost'
        },
        'custom_model': {
            0: 'ball',      # → 'football'
            1: 'marker',    # → 'cone'
            2: 'person',    # → 'player'
            3: 'goal'       # → 'goalpost'
        }
    },
    'class_aliases': {
        'ball': 'football',
        'marker': 'cone',
        'person': 'player',
        'goal': 'goalpost'
    }
}
```

### Traffic Detection
```python
'traffic_detection': {
    'standard_classes': ['car', 'truck', 'bus', 'motorcycle', 'bicycle', 'person'],
    'model_mappings': {
        'coco_traffic': {
            2: 'car',
            7: 'truck',
            5: 'bus',
            3: 'motorcycle',
            1: 'bicycle',
            0: 'person'
        }
    },
    'class_aliases': {
        'vehicle': 'car',
        'motorbike': 'motorcycle',
        'bike': 'bicycle'
    }
}
```

## API Endpoints

### Get Current Configuration
```
GET /get_class_mappings
```
Returns current mapping configuration and selected classes.

### Update Configuration
```
POST /set_class_mapping
Content-Type: application/json

{
    "mapping_name": "football_detection",
    "selected_classes": ["football", "cone"]
}
```

### Upload and Analyze
```
POST /upload
Content-Type: multipart/form-data

files[]: [uploaded files]
media_type: "images" or "videos"
```

## Advanced Features

### 1. Model Type Auto-Detection
The system attempts to automatically detect model types based on filename patterns:
- `yolov8_football.pt` → Football detection mapping
- `coco_trained.pt` → COCO mapping
- `traffic_model.pt` → Traffic detection mapping

### 2. Confidence Filtering
Results are filtered to only show detections for selected classes, ensuring fair comparison.

### 3. Flexible Aliases
The alias system handles various naming conventions:
- "ball" → "football"
- "person" → "player"
- "vehicle" → "car"

### 4. Performance Metrics
- **Detection Count**: Number of objects detected per class
- **Confidence Scores**: Average confidence per class
- **Speed Metrics**: Inference time comparison
- **Overall Winner**: Determined by multiple factors

## Troubleshooting

### Common Issues

1. **"Model files not found"**
   - Ensure model files are in the `models/` directory
   - Check file extensions (.pt or .onnx)

2. **"No detections found"**
   - Verify class mapping is correct
   - Check if selected classes exist in model output
   - Lower confidence threshold if needed

3. **"Configuration not applied"**
   - Ensure at least one class is selected
   - Check browser console for errors
   - Refresh page and try again

### Debug Mode
Enable debug output by modifying the class mapping functions to print intermediate results:

```python
def map_model_classes(model, mapping_config, selected_classes):
    print(f"Model classes: {model.names}")
    print(f"Selected classes: {selected_classes}")
    # ... rest of function
```

## Customization

### Adding New Detection Types
1. Add a new entry to `CLASS_MAPPINGS`
2. Define standard classes and model mappings
3. Include relevant aliases
4. Update the web interface if needed

### Modifying Comparison Logic
Edit the `compare_models()` function to adjust how models are ranked:
- Weight different metrics differently
- Add new comparison criteria
- Customize scoring system

### Styling Changes
Modify the CSS in `index.html` to customize:
- Color schemes
- Layout
- Typography
- Animations

## Future Enhancements

- **Multi-model Support**: Compare more than 2 models
- **Custom Metrics**: User-defined comparison criteria
- **Export Results**: Save comparisons as reports
- **Model Training Integration**: Direct training feedback
- **Advanced Filtering**: Confidence thresholds, size filters

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add your enhancements
4. Test thoroughly
5. Submit a pull request

## License

This project is open source and available under the MIT License.