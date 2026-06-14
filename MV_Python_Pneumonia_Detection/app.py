"""
Flask Backend for Pneumonia Detection Web UI
Integrates with the existing Grad-CAM detection script
"""

import os
import base64
import io
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

app = Flask(__name__, static_folder='static')
CORS(app)

# ── Configuration ────────────────────────────────────────────────────────────
MODELS_DIR = r"D:/Projects/MV_Python/MV_Python_Pneumonia_Detection/Models"
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

IMG_SIZE = 224

MODEL_WEIGHTS = {
    "cnn": 0.15,
    "mobilenet": 0.30,
    "resnet": 0.25,
    "mobilenet_ft": 0.30,
}

ENSEMBLE_THRESHOLD = 0.50
BASE_THRESHOLD = 0.60
GRADCAM_ALPHA = 0.45

MODEL_FILES = {
    "cnn": "cnn_model.h5",
    "mobilenet": "MobileNet_model.h5",
    "resnet": "model_transferLearn_RestNet50.h5",
    "mobilenet_ft": "FineTuneMobileNet.h5",
}

MODEL_PIPELINE = {
    "cnn": "cnn",
    "mobilenet": "transfer",
    "resnet": "transfer",
    "mobilenet_ft": "transfer",
}

LAST_CONV = {
    "cnn": "conv2d_2",
    "mobilenet": "Conv_1",
    "resnet": "conv5_block3_out",
    "mobilenet_ft": "Conv_1",
}

MODEL_LABELS = {
    "cnn": "Custom CNN",
    "mobilenet": "MobileNetV2",
    "resnet": "ResNet50",
    "mobilenet_ft": "MobileNet Fine-Tuned",
}

# Global models storage
models = {}


# ── Helper Functions ─────────────────────────────────────────────────────────
def img_to_base64(img_array):
    """Convert numpy array to base64 string"""
    if img_array.dtype != np.uint8:
        img_array = (np.clip(img_array, 0, 1) * 255).astype(np.uint8)
    
    if len(img_array.shape) == 2:
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
    elif img_array.shape[2] == 3:
        img_bgr = img_array
    else:
        img_bgr = img_array[:, :, :3]
    
    _, buffer = cv2.imencode('.png', img_bgr)
    return base64.b64encode(buffer).decode('utf-8')


def _clean_mask(mask):
    k = np.ones((5, 5), np.uint8)
    return cv2.morphologyEx(cv2.morphologyEx(mask, cv2.MORPH_OPEN, k), cv2.MORPH_CLOSE, k)


def _best_contour(contours):
    big = [c for c in contours if cv2.contourArea(c) > 500]
    return max(big, key=cv2.contourArea) if big else None


def _register(img):
    M = np.float32([[1, 0, 5], [0, 1, 5]])
    return cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))


# ── Preprocessing with step tracking ────────────────────────────────────────
def process_cnn_pipeline(img_path):
    """CNN pipeline with step-by-step outputs"""
    steps = []
    
    # Step 1: Load original
    img_original = cv2.imread(img_path)
    img_resized = cv2.resize(img_original, (IMG_SIZE, IMG_SIZE))
    steps.append({
        "step": 1,
        "name": "Original X-Ray",
        "description": "Raw chest X-ray image loaded and resized to 224×224 pixels for processing.",
        "image": img_to_base64(cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB))
    })
    
    # Step 2: Gaussian Blur
    img_blurred = cv2.GaussianBlur(img_resized, (5, 5), 0)
    steps.append({
        "step": 2,
        "name": "Gaussian Blur",
        "description": "Applied 5×5 Gaussian filter to reduce noise and smooth the image while preserving edges.",
        "image": img_to_base64(cv2.cvtColor(img_blurred, cv2.COLOR_BGR2RGB))
    })
    
    # Step 3: Thresholding
    gray = cv2.cvtColor(img_blurred, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    steps.append({
        "step": 3,
        "name": "Binary Thresholding",
        "description": "Converted to binary image (threshold=127) to separate lung region from background.",
        "image": img_to_base64(thresh)
    })
    
    # Step 4: Morphological operations
    mask = _clean_mask(thresh)
    steps.append({
        "step": 4,
        "name": "Morphological Cleaning",
        "description": "Applied opening and closing operations to remove noise and fill gaps in the mask.",
        "image": img_to_base64(mask)
    })
    
    # Step 5: Segmentation
    segmented = cv2.bitwise_and(img_blurred, img_blurred, mask=mask)
    steps.append({
        "step": 5,
        "name": "Lung Segmentation",
        "description": "...",
        "image": img_to_base64(cv2.cvtColor(segmented, cv2.COLOR_BGR2RGB))
    })

# 👉 FIND CONTOUR FIRST
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = _best_contour(contours)

    if best is not None:
        x, y, w, h = cv2.boundingRect(best)

        # ✅ Step 5.5 (NOW SAFE)
        vis = img_blurred.copy()
        cv2.rectangle(vis, (x, y), (x+w, y+h), (0,255,0), 2)

        steps.append({
            "step": 5.5,
            "name": "ROI Bounding Box",
            "description": "Detected lung region highlighted before cropping.",
            "image": img_to_base64(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
        })

        # ✅ ROI with padding
        pad = 20
        x1 = max(x - pad, 0)
        y1 = max(y - pad, 0)
        x2 = min(x + w + pad, segmented.shape[1])
        y2 = min(y + h + pad, segmented.shape[0])

        roi = segmented[y1:y2, x1:x2]
        roi_resized = cv2.resize(roi, (IMG_SIZE, IMG_SIZE))

    else:
        roi_resized = segmented
    steps.append({
        "step": 6,
        "name": "ROI Extraction",
        "description": "Detected largest contour and cropped to bounding box, then resized to standard dimensions.",
        "image": img_to_base64(cv2.cvtColor(roi_resized, cv2.COLOR_BGR2RGB))
    })
    
    # Step 7: Registration (shift)
    registered = _register(roi_resized)
    steps.append({
        "step": 7,
        "name": "Image Registration",
        "description": "Applied affine transformation (5-pixel shift) for spatial alignment augmentation.",
        "image": img_to_base64(cv2.cvtColor(registered, cv2.COLOR_BGR2RGB))
    })
    
    # Step 8: Normalization
    normalized = cv2.resize(registered, (IMG_SIZE, IMG_SIZE)).astype(np.float32) / 255.0
    if normalized.ndim == 2:
        normalized = np.stack([normalized] * 3, axis=-1)
    steps.append({
        "step": 8,
        "name": "Normalization",
        "description": "Normalized pixel values to [0,1] range for neural network input. Ready for CNN model.",
        "image": img_to_base64((normalized * 255).astype(np.uint8))
    })
    
    return steps, np.expand_dims(normalized, 0)


def process_transfer_pipeline(img_path):
    """Transfer learning pipeline with step-by-step outputs"""
    steps = []
    
    # Step 1: Load original
    img_original = cv2.imread(img_path)
    img_resized = cv2.resize(img_original, (IMG_SIZE, IMG_SIZE))
    steps.append({
        "step": 1,
        "name": "Original X-Ray",
        "description": "Raw chest X-ray image loaded for transfer learning preprocessing.",
        "image": img_to_base64(cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB))
    })
    
    # Step 2: Thresholding
    gray = cv2.cvtColor(img_original, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    steps.append({
        "step": 2,
        "name": "Binary Thresholding",
        "description": "Separated foreground from background using binary threshold (value=127).",
        "image": img_to_base64(thresh)
    })
    
    # Step 3: Morphological operations
    k = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    steps.append({
        "step": 3,
        "name": "Morphological Refinement",
        "description": "Cleaned mask using opening (removes small noise) and closing (fills small holes).",
        "image": img_to_base64(mask)
    })
    
    # Step 4: Segmentation
    segmented = cv2.bitwise_and(img_original, img_original, mask=mask)
    steps.append({
        "step": 4,
        "name": "Segmented Region",
        "description": "Applied mask to extract lung region from original image.",
        "image": img_to_base64(cv2.cvtColor(segmented, cv2.COLOR_BGR2RGB))
    })
    
    # Step 5: ROI detection (with bounding box)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = _best_contour(contours)

    if best is not None:
        x, y, w, h = cv2.boundingRect(best)

        # ✅ Draw box on ORIGINAL (important for clarity)
        vis = img_original.copy()
        cv2.rectangle(vis, (x, y), (x+w, y+h), (0,255,0), 2)

        steps.append({
            "step": 4.5,
            "name": "ROI Bounding Box",
            "description": "Detected lung region highlighted before cropping.",
            "image": img_to_base64(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
        })

        # ✅ ROI with padding (same as CNN)
        pad = 20
        x1 = max(x - pad, 0)
        y1 = max(y - pad, 0)
        x2 = min(x + w + pad, segmented.shape[1])
        y2 = min(y + h + pad, segmented.shape[0])

        roi = segmented[y1:y2, x1:x2]
        roi_resized = cv2.resize(roi, (IMG_SIZE, IMG_SIZE))

    else:
        roi_resized = cv2.resize(segmented, (IMG_SIZE, IMG_SIZE))
    steps.append({
        "step": 5,
        "name": "ROI Cropping",
        "description": "Extracted region of interest by finding largest contour and resizing to 224×224.",
        "image": img_to_base64(cv2.cvtColor(roi_resized, cv2.COLOR_BGR2RGB))
    })
    
    # Step 6: Normalization
    normalized = roi_resized.astype(np.float32) / 255.0
    if normalized.ndim == 2:
        normalized = np.stack([normalized] * 3, axis=-1)
    steps.append({
        "step": 6,
        "name": "Normalization",
        "description": "Normalized to [0,1] range. Ready for MobileNet/ResNet transfer learning models.",
        "image": img_to_base64((normalized * 255).astype(np.uint8))
    })
    
    return steps, np.expand_dims(normalized, 0)


# ── Grad-CAM ─────────────────────────────────────────────────────────────────
def _find_last_conv(model):
    for layer in reversed(model.layers):
        if isinstance(layer, (tf.keras.layers.Conv2D, tf.keras.layers.DepthwiseConv2D)):
            return layer.name
    return None


def _warmup(model, img_array):
    try:
        model(img_array, training=False)
    except Exception:
        pass


def make_gradcam_heatmap(model, img_array, last_conv_name):
    _warmup(model, img_array)
    
    try:
        model.get_layer(last_conv_name)
    except ValueError:
        last_conv_name = _find_last_conv(model)
        if last_conv_name is None:
            return None
    
    grad_model = None
    try:
        grad_model = Model(
            inputs=model.inputs,
            outputs=[model.get_layer(last_conv_name).output, model.output]
        )
    except Exception:
        try:
            inp_tensor = tf.keras.Input(shape=img_array.shape[1:])
            x = inp_tensor
            conv_out_tensor = None
            for layer in model.layers:
                x = layer(x)
                if layer.name == last_conv_name:
                    conv_out_tensor = x
            if conv_out_tensor is None:
                return None
            grad_model = Model(inputs=inp_tensor, outputs=[conv_out_tensor, x])
        except Exception:
            return None
    
    if grad_model is None:
        return None
    
    img_tensor = tf.cast(img_array, tf.float32)
    
    with tf.GradientTape() as tape:
        outputs = grad_model(img_tensor, training=False)
        if isinstance(outputs, (list, tuple)) and len(outputs) == 2:
            conv_outputs = outputs[0]
            predictions = outputs[1]
        else:
            return None
        
        tape.watch(conv_outputs)
        
        if not isinstance(predictions, tf.Tensor):
            try:
                predictions = tf.convert_to_tensor(predictions)
            except Exception:
                return None
        
        class_channel = predictions[:, 0]
    
    grads = tape.gradient(class_channel, conv_outputs)
    if grads is None:
        return None
    
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_out = conv_outputs[0]
    heatmap = conv_out @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    
    if heatmap.shape.rank == 0 or tf.size(heatmap) < 2:
        return None
    
    heatmap = tf.maximum(heatmap, 0)
    max_val = tf.reduce_max(heatmap)
    heatmap = heatmap / (max_val + 1e-8)
    return heatmap.numpy()


def overlay_gradcam(img_bgr_uint8, heatmap, alpha=GRADCAM_ALPHA):
    h, w = img_bgr_uint8.shape[:2]
    hm = cv2.resize(heatmap, (w, h))
    hm = np.uint8(255 * np.clip(hm, 0, 1))
    hm = cv2.applyColorMap(hm, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img_bgr_uint8, 1 - alpha, hm, alpha, 0)
    return cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)


# ── Model Loading ────────────────────────────────────────────────────────────
def load_all_models():
    global models
    for key, fname in MODEL_FILES.items():
        path = os.path.join(MODELS_DIR, fname)
        if os.path.isfile(path):
            print(f"✓ Loading {fname}")
            models[key] = load_model(path, compile=False)
        else:
            print(f"✗ Not found: {fname}")
            models[key] = None


# ── API Routes ───────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return send_from_directory('static', 'index.html')


@app.route('/api/process', methods=['POST'])
def process_image():
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400
    
    file = request.files['image']
    filepath = os.path.join(UPLOAD_DIR, file.filename)
    file.save(filepath)
    
    try:
        # Process both pipelines
        cnn_steps, cnn_input = process_cnn_pipeline(filepath)
        transfer_steps, transfer_input = process_transfer_pipeline(filepath)
        
        # Run predictions
        preds = {}
        for key in ["cnn", "mobilenet", "resnet", "mobilenet_ft"]:
            if models[key] is not None:
                pipe = MODEL_PIPELINE[key]
                inp = cnn_input if pipe == "cnn" else transfer_input
                preds[key] = float(models[key].predict(inp, verbose=0)[0][0])
            else:
                preds[key] = None
        
        # Calculate ensemble
        available_keys = [k for k in preds if preds[k] is not None]
        weight_sum = sum(MODEL_WEIGHTS[k] for k in available_keys)
        weighted_score = sum(MODEL_WEIGHTS[k] * preds[k] for k in available_keys) / weight_sum
        
        votes_pneumonia = sum(1 for k in available_keys if preds[k] > BASE_THRESHOLD)
        majority_label = "PNEUMONIA" if votes_pneumonia > len(available_keys) / 2 else "NORMAL"
        
        if weighted_score > ENSEMBLE_THRESHOLD:
            final_label = "PNEUMONIA"
        elif weighted_score < ENSEMBLE_THRESHOLD:
            final_label = "NORMAL"
        else:
            final_label = majority_label
        
        confidence = weighted_score if final_label == "PNEUMONIA" else 1.0 - weighted_score
        
        # Generate Grad-CAM visualizations
        gradcam_results = []
        inputs_dict = {"cnn": cnn_input, "transfer": transfer_input}
        
        for key in ["cnn", "mobilenet", "resnet", "mobilenet_ft"]:
            if models[key] is None or preds[key] is None:
                continue
            
            pipe = MODEL_PIPELINE[key]
            inp = inputs_dict[pipe]
            hm = make_gradcam_heatmap(models[key], inp, LAST_CONV[key])
            
            if hm is not None:
                src_bgr = (inp[0] * 255).astype(np.uint8)
                overlay = overlay_gradcam(src_bgr, hm)
                
                label = "PNEUMONIA" if preds[key] > BASE_THRESHOLD else "NORMAL"
                
                gradcam_results.append({
                    "model": MODEL_LABELS[key],
                    "probability": round(preds[key], 3),
                    "label": label,
                    "image": img_to_base64(overlay)
                })
        
        # Prepare model predictions for display
        model_predictions = []
        for key in ["cnn", "mobilenet", "resnet", "mobilenet_ft"]:
            if preds[key] is not None:
                model_predictions.append({
                    "model": MODEL_LABELS[key],
                    "probability": round(preds[key], 3),
                    "label": "PNEUMONIA" if preds[key] > BASE_THRESHOLD else "NORMAL",
                    "weight": MODEL_WEIGHTS[key]
                })
        
        response = {
            "cnn_pipeline": cnn_steps,
            "transfer_pipeline": transfer_steps,
            "predictions": model_predictions,
            "gradcam": gradcam_results,
            "ensemble": {
                "weighted_score": round(weighted_score, 3),
                "final_label": final_label,
                "confidence": round(confidence * 100, 1),
                "votes_pneumonia": votes_pneumonia,
                "total_models": len(available_keys)
            }
        }
        
        return jsonify(response)
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(filepath):
            os.remove(filepath)


if __name__ == '__main__':
    print("Loading models...")
    load_all_models()
    print("Starting server...")
    app.run(debug=True, port=5000)