YOLO Model Comparison Tool
This Flask application provides a web interface to upload images and compare the performance of two different YOLOv8 models for object detection. It is specifically tailored to analyze detections for 'football' and 'cone' classes but can be adapted for others.

Features
Web-Based UI: Simple and clean interface for uploading multiple images.

Side-by-Side Comparison: View the output of both models on each image. Bounding boxes from Model 1 are green, and Model 2 are blue.

Detailed Analytics: For each image, see the number of detections and average confidence score for each model.

Performance Winner: A simple algorithm determines which model performed "better" on a per-image basis.

Overall Summary: Get aggregate statistics across all uploaded images, including total detections, overall average confidence, and the number of images "won" by each model.

Project Structure
/
|-- app.py                  # The main Flask application
|-- requirements.txt        # Python dependencies
|-- README.md               # This file
|
|-- models/                 # FOLDER: Place your YOLO models here
|   |-- best_model_1.pt     # Your first model
|   `-- best_model_2.pt     # Your second model
|
|-- templates/              # FOLDER: Contains HTML templates
|   `-- index.html          # The main web page
|
|-- static/                 # FOLDER: Will be created automatically
    |-- uploads/            # For original uploaded images
    `-- processed/          # For images with bounding boxes

Setup and Installation
Clone the repository or download the files.

Create a Python virtual environment (recommended):

python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

Install the required dependencies:

pip install -r requirements.txt

Place your models:

Create a folder named models in the root directory.

Place your two trained YOLOv8 model files (.pt) inside the models folder.

Crucially, you must rename them to best_model_1.pt and best_model_2.pt.

Run the Flask application:

flask run

Or for development mode:

python app.py

Access the application:
Open your web browser and navigate to http://127.0.0.1:5000.

How to Use
Navigate to the web interface.

Click the "Upload" area or drag and drop your test images (PNG, JPG, JPEG).

Once files are selected, click the "Analyze Performance" button.

Wait for the analysis to complete. The time taken will depend on the number of images and the complexity of the models.

Review the results, which are broken down into an overall summary and a detailed card for each image.

How the Comparison Works
The application determines a "better" model for each image based on a simple scoring system:

+1 point to the model with more total detections.

+1 point to the model with a higher average confidence score across all detections.

+0.5 points to the model that detects more of each specific class ('football', 'cone').

The model with the higher total score is declared the winner for that image. This is a basic heuristic and does not replace a rigorous evaluation with a labeled ground truth dataset, but it provides a quick and useful first-pass comparison.

