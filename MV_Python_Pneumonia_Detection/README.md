# Pneumonia Detection System - Web Interface

A professional medical-grade web interface for pneumonia detection using deep learning with step-by-step preprocessing visualization and Grad-CAM explanations.

## Features

✨ **Clean Medical UI** - Light, professional theme suitable for clinical presentations
🔬 **Step-by-Step Visualization** - See every preprocessing step with clear explanations
🤖 **4 Deep Learning Models** - Custom CNN, MobileNet, ResNet50, and Fine-tuned MobileNet
🎯 **Grad-CAM Heatmaps** - Visual explanations of model attention
📊 **Ensemble Prediction** - Confidence-weighted voting with majority consensus
🎓 **Perfect for Viva** - Clear process flow for academic presentations

## Setup Instructions

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Update Model Paths

Open `app.py` and update line 17 with your models directory:

```python
MODELS_DIR = r"D:/Projects/MV_Python/MV_Python_Pneumonia_Detection/Models"
```

Make sure your Models folder contains these files:
- `cnn_model.h5`
- `MobileNet_model.h5`
- `model_transferLearn_RestNet50.h5`
- `FineTuneMobileNet.h5`

### 3. Run the Application

```bash
python app.py
```

The server will start on `http://localhost:5000`

### 4. Open in Browser

Navigate to `http://localhost:5000` in your web browser.

## How to Use

1. **Upload X-Ray**: Click "Choose X-Ray Image" or drag & drop an image
2. **Wait for Processing**: The system will analyze with all 4 models (~10-15 seconds)
3. **View Results**: 
   - **Section 1**: CNN preprocessing steps (8 steps with explanations)
   - **Section 2**: Transfer learning preprocessing (6 steps)
   - **Section 3**: Individual model predictions with probabilities
   - **Section 4**: Grad-CAM attention maps showing focus regions
   - **Top**: Final ensemble diagnosis with confidence

## For Viva Presentation

The interface is designed to clearly show:

### CNN Pipeline (Section 1)
1. **Original X-Ray** - Raw input image
2. **Gaussian Blur** - Noise reduction
3. **Binary Thresholding** - Foreground/background separation
4. **Morphological Cleaning** - Noise removal
5. **Lung Segmentation** - ROI extraction
6. **ROI Extraction** - Bounding box detection
7. **Image Registration** - Spatial alignment
8. **Normalization** - Neural network preparation

### Transfer Learning Pipeline (Section 2)
1. **Original X-Ray** - Raw input
2. **Binary Thresholding** - Segmentation
3. **Morphological Refinement** - Mask cleaning
4. **Segmented Region** - Masked extraction
5. **ROI Cropping** - Contour-based cropping
6. **Normalization** - Model-ready input

### Ensemble Logic
- Each model outputs probability [0,1]
- Weighted average using model performance weights:
  - CNN: 0.15 (custom, weakest)
  - MobileNet: 0.30 (frozen transfer)
  - ResNet50: 0.25 (transfer learning)
  - MobileNet FT: 0.30 (fine-tuned, strongest)
- Threshold: 0.50 for final classification
- Majority vote as tiebreaker

## Technical Details

### Backend (Flask)
- `app.py` - Main Flask application
- Processes images through both pipelines
- Generates step-by-step visualizations
- Runs inference on all 4 models
- Computes Grad-CAM heatmaps
- Returns JSON with base64-encoded images

### Frontend (HTML/CSS/JS)
- `static/index.html` - Single-page application
- Clean medical-grade design
- Responsive layout
- Animated step reveals
- Real-time progress feedback

### Color Scheme
- Primary: Medical blue (#0066CC)
- Success: Clinical green (#00875A)
- Danger: Alert red (#DE350B)
- Background: Clean white/light gray
- Typography: IBM Plex Sans (professional, readable)

## Troubleshooting

**Models not loading?**
- Check MODELS_DIR path in app.py
- Verify all 4 .h5 files exist
- Ensure TensorFlow version compatibility

**Image processing fails?**
- Ensure image is valid chest X-ray
- Check file format (JPEG, PNG supported)
- Verify image is not corrupted

**Slow processing?**
- Normal: 10-15 seconds for 4 models
- Use GPU if available (check TensorFlow GPU setup)
- Consider reducing image size if needed

## File Structure

```
project/
├── app.py                      # Flask backend
├── requirements.txt            # Python dependencies
├── static/
│   └── index.html             # Web interface
├── uploads/                   # Temporary upload folder (auto-created)
└── README.md                  # This file
```

## Notes for Faculty

**Why confidence-weighted ensemble?**
- Meta model showed only ~80% accuracy (overfitting)
- Weighted ensemble uses model confidence automatically
- Majority vote prevents edge-case errors
- More robust than single meta-learner

**Grad-CAM Benefits**
- Shows interpretability (not black box)
- Validates model attention on lung regions
- Helps identify potential issues
- Aligns with medical diagnosis workflow

## License

Academic use only. For viva presentation and educational purposes.