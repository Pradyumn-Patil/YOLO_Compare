# 🏈 YOLO Model Comparison Tool

<div align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-blue.svg" alt="Python Version">
  <img src="https://img.shields.io/badge/Flask-2.0+-green.svg" alt="Flask Version">
  <img src="https://img.shields.io/badge/YOLOv8-Ultralytics-orange.svg" alt="YOLO Version">
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License">
</div>

<br>

A **powerful Flask web application** that provides an intuitive interface to upload images and compare the performance of two different YOLOv8 models for object detection. Specifically tailored for analyzing **'football'** and **'cone'** detections, but easily adaptable for other object classes.

---

## ✨ Features

### 🖥️ **Web-Based Interface**
- Clean, modern UI for uploading multiple images
- Drag-and-drop functionality for seamless file uploads
- Real-time progress tracking during analysis

### 🔄 **Side-by-Side Comparison**
- Visual comparison of both models on each image
- **Color-coded bounding boxes**: 
  - 🟢 **Green** for Model 1 detections
  - 🔵 **Blue** for Model 2 detections

### 📊 **Detailed Analytics**
- **Per-image statistics**: Detection counts and confidence scores
- **Class-specific analysis**: Individual metrics for each object type
- **Speed benchmarks**: Preprocessing, inference, and postprocessing times

### 🏆 **Performance Winner**
- Intelligent algorithm determines the better-performing model
- **Scoring system** based on detection accuracy and confidence
- Per-image and overall performance rankings

### 📈 **Comprehensive Summary**
- **Aggregate statistics** across all uploaded images
- Total detections and average confidence scores
- Model comparison with detailed breakdowns

---

## 📁 Project Structure

```
📦 YOLO-Model-Comparison-Tool/
├── 🐍 app.py                  # Main Flask application
├── 📋 requirements.txt        # Python dependencies
├── 📖 README.md               # Documentation
├── 
├── 📁 models/                 # 🔴 IMPORTANT: Place your models here
│   ├── 🎯 best_model_1.pt     # Your first YOLO model
│   └── 🎯 best_model_2.pt     # Your second YOLO model
├── 
├── 📁 templates/              # HTML templates
│   └── 🌐 index.html          # Main web interface
├── 
└── 📁 static/                 # Auto-generated folders
    ├── 📁 uploads/            # Original uploaded images
    └── 📁 processed/          # Images with bounding boxes
```

---

## 🚀 Setup and Installation

### 1️⃣ **Get the Code**
```bash
git clone <repository-url>
cd MODEL_TEST_APP
```

### 2️⃣ **Create Virtual Environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3️⃣ **Install Dependencies**
```bash
pip install -r requirements.txt
```

### 4️⃣ **Prepare Your Models**
- Create a `models/` folder in the root directory
- Place your trained YOLOv8 model files (`.pt`) inside
- **⚠️ IMPORTANT**: Rename them to:
  - `best_model_1.pt`
  - `best_model_2.pt`

### 5️⃣ **Launch the Application**
```bash
# Development mode
python app.py

# Or using Flask command
flask run
```

### 6️⃣ **Access the Interface**
Open your browser and navigate to: **http://127.0.0.1:5000**

---

## 🎯 How to Use

| Step | Action | Description |
|------|--------|-------------|
| 1️⃣ | **Upload** | Navigate to the web interface and upload your test images |
| 2️⃣ | **Select** | Choose multiple images (PNG, JPG, JPEG) via drag-drop or click |
| 3️⃣ | **Analyze** | Click "Analyze Performance" to start the comparison |
| 4️⃣ | **Wait** | Processing time depends on image count and model complexity |
| 5️⃣ | **Review** | Examine detailed results and overall performance summary |

---

## 🏆 Comparison Methodology

Our intelligent scoring system evaluates models based on:

### 📊 **Scoring Criteria**

| Metric | Points | Description |
|--------|--------|-------------|
| 🎯 **Total Detections** | +1.0 | Model with more overall detections |
| 🎖️ **Average Confidence** | +1.0 | Model with higher confidence scores |
| ⚡ **Inference Speed** | +1.0 | Model with faster processing time |
| 🏈 **Football Detections** | +0.5 | Model detecting more footballs |
| 🚧 **Cone Detections** | +0.5 | Model detecting more cones |

### 🏅 **Winner Determination**
- Model with **highest total score** wins each image
- **Tie-breaker**: Similar performance noted
- **Overall champion**: Model winning most images

> 💡 **Note**: This is a heuristic evaluation tool and doesn't replace rigorous testing with labeled ground truth datasets.

---

## 📊 Performance Metrics

The application tracks and displays:

- ⏱️ **Processing Speed**: Preprocess, inference, and postprocess times
- 🎯 **Detection Accuracy**: Count and confidence for each class
- 📈 **Comparative Analysis**: Side-by-side model performance
- 🏆 **Winner Statistics**: Per-image and overall rankings

---

## 🔧 Technical Requirements

- **Python**: 3.8 or higher
- **Flask**: 2.0+
- **OpenCV**: For image processing
- **Ultralytics**: For YOLO model support
- **NumPy**: For numerical operations

---

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

<div align="center">
  <p>Made with ❤️ for computer vision enthusiasts</p>
  <p>⭐ Star this repo if you find it useful!</p>
</div>