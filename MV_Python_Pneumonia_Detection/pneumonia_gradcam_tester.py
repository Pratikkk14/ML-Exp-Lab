"""
=============================================================
  Pneumonia Detection — Single Image Tester with Grad-CAM
=============================================================
Expected filenames (exact match):
  cnn_model.h5
  MobileNet_model.h5
  model_transferLearn_RestNet50.h5
  FineTuneMobileNet.h5

=============================================================
"""

# ── USER CONFIG ──────────────────────────────────────────────────────────────

MODELS_DIR = r"D:/Projects/MV_Python/MV_Python_Pneumonia_Detection/Models"
OUTPUT_DIR = r"D:/Projects/MV_Python/MV_Python_Pneumonia_Detection/Output"

MODEL_WEIGHTS = {
    "cnn":          0.15,
    "mobilenet":    0.30,
    "resnet":       0.25,
    "mobilenet_ft": 0.30,
}

ENSEMBLE_THRESHOLD = 0.50
BASE_THRESHOLD     = 0.60
GRADCAM_ALPHA      = 0.45

# ─────────────────────────────────────────────────────────────────────────────

import os
import sys
import warnings
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import tensorflow as tf
from tensorflow.keras.models import load_model, Model

IMG_SIZE = 224

# ─────────────────────────────────────────────────────────────────────────────
# PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────

def _clean_mask(mask):
    k = np.ones((5, 5), np.uint8)
    return cv2.morphologyEx(cv2.morphologyEx(mask, cv2.MORPH_OPEN, k), cv2.MORPH_CLOSE, k)


def _best_contour(contours):
    big = [c for c in contours if cv2.contourArea(c) > 500]
    return max(big, key=cv2.contourArea) if big else None


def segment_cnn(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    mask = _clean_mask(thresh)
    segmented = cv2.bitwise_and(img_bgr, img_bgr, mask=mask)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = _best_contour(contours)
    if best is None:
        return cv2.resize(img_bgr, (IMG_SIZE, IMG_SIZE))
    x, y, w, h = cv2.boundingRect(best)
    roi = segmented[y:y+h, x:x+w]
    return cv2.resize(roi, (IMG_SIZE, IMG_SIZE))


def segment_transfer(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    k = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    segmented = cv2.bitwise_and(img_bgr, img_bgr, mask=mask)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = _best_contour(contours)
    if best is None:
        return cv2.resize(img_bgr, (IMG_SIZE, IMG_SIZE))
    x, y, w, h = cv2.boundingRect(best)
    roi = segmented[y:y+h, x:x+w]
    return cv2.resize(roi, (IMG_SIZE, IMG_SIZE))


def _register(img):
    M = np.float32([[1, 0, 5], [0, 1, 5]])
    return cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))


def make_cnn_input(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = cv2.GaussianBlur(img, (5, 5), 0)
    roi = segment_cnn(img)
    reg = _register(roi)
    out = cv2.resize(reg, (IMG_SIZE, IMG_SIZE)).astype(np.float32) / 255.0
    if out.ndim == 2:
        out = np.stack([out] * 3, axis=-1)
    return np.expand_dims(out, 0)


def make_transfer_input(img_path):
    img = cv2.imread(img_path)
    roi = segment_transfer(img)
    out = roi.astype(np.float32) / 255.0
    if out.ndim == 2:
        out = np.stack([out] * 3, axis=-1)
    return np.expand_dims(out, 0)


# ─────────────────────────────────────────────────────────────────────────────
# GRAD-CAM
# ─────────────────────────────────────────────────────────────────────────────

LAST_CONV = {
    "cnn":          "conv2d_2",
    "mobilenet":    "Conv_1",
    "resnet":       "conv5_block3_out",
    "mobilenet_ft": "Conv_1",
}


def _find_last_conv(model):
    for layer in reversed(model.layers):
        if isinstance(layer, (tf.keras.layers.Conv2D,
                               tf.keras.layers.DepthwiseConv2D)):
            return layer.name
    return None


def _warmup(model, img_array):
    try:
        model(img_array, training=False)
    except Exception:
        pass


def make_gradcam_heatmap(model, img_array, last_conv_name):
    """
    Compute Grad-CAM heatmap.
    Returns float32 array shape (H, W) normalised [0,1], or None on failure.
    """
    _warmup(model, img_array)

    # Resolve conv layer name
    try:
        model.get_layer(last_conv_name)
    except ValueError:
        last_conv_name = _find_last_conv(model)
        if last_conv_name is None:
            return None

    # Build grad model — try functional shortcut first, then layer-by-layer
    grad_model = None
    try:
        grad_model = Model(
            inputs  = model.inputs,
            outputs = [model.get_layer(last_conv_name).output, model.output]
        )
    except Exception:
        try:
            inp_tensor      = tf.keras.Input(shape=img_array.shape[1:])
            x               = inp_tensor
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

    # ── GradientTape pass ────────────────────────────────────────────────────
    img_tensor = tf.cast(img_array, tf.float32)

    with tf.GradientTape() as tape:
        # Call the model and unpack outputs
        outputs = grad_model(img_tensor, training=False)
        
        # Handle both tuple/list returns - ensure we get tensors
        if isinstance(outputs, (list, tuple)) and len(outputs) == 2:
            conv_outputs = outputs[0]
            predictions = outputs[1]
        else:
            return None
        
        # Watch the conv outputs for gradient computation
        tape.watch(conv_outputs)
        
        # Ensure predictions is a tensor (handle nested lists from Sequential models)
        if not isinstance(predictions, tf.Tensor):
            try:
                predictions = tf.convert_to_tensor(predictions)
            except Exception:
                return None
        
        # Extract the class prediction (assumes binary classification, single output neuron)
        if predictions.shape[-1] == 1:
            class_channel = predictions[:, 0]
        else:
            class_channel = predictions[:, 0]  # Take first output

    grads = tape.gradient(class_channel, conv_outputs)
    if grads is None:
        return None

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))   # (C,)
    conv_out     = conv_outputs[0]                          # (H, W, C)

    # Weighted sum of feature maps
    heatmap = conv_out @ pooled_grads[..., tf.newaxis]      # (H, W, 1)
    heatmap = tf.squeeze(heatmap)                           # (H, W) or scalar

    # Guard against degenerate shapes
    if heatmap.shape.rank == 0 or tf.size(heatmap) < 2:
        return None

    heatmap  = tf.maximum(heatmap, 0)
    max_val  = tf.reduce_max(heatmap)
    heatmap  = heatmap / (max_val + 1e-8)
    return heatmap.numpy()


def overlay_gradcam(img_bgr_uint8, heatmap, alpha=GRADCAM_ALPHA):
    """
    Blend JET-coloured heatmap onto a uint8 BGR image.
    Returns RGB uint8 (H, W, 3).
    """
    h, w = img_bgr_uint8.shape[:2]
    hm   = cv2.resize(heatmap, (w, h))
    hm   = np.uint8(255 * np.clip(hm, 0, 1))
    hm   = cv2.applyColorMap(hm, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img_bgr_uint8, 1 - alpha, hm, alpha, 0)
    return cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)


# ─────────────────────────────────────────────────────────────────────────────
# MODEL LOADING
# ─────────────────────────────────────────────────────────────────────────────

MODEL_FILES = {
    "cnn":          "cnn_model.h5",
    "mobilenet":    "MobileNet_model.h5",
    "resnet":       "model_transferLearn_RestNet50.h5",
    "mobilenet_ft": "FineTuneMobileNet.h5",
}

MODEL_PIPELINE = {
    "cnn":          "cnn",
    "mobilenet":    "transfer",
    "resnet":       "transfer",
    "mobilenet_ft": "transfer",
}


def load_all_models(models_dir):
    loaded = {}
    print()
    for key, fname in MODEL_FILES.items():
        path = os.path.join(models_dir, fname)
        if os.path.isfile(path):
            print(f"  ✓  Loading  {fname}")
            loaded[key] = load_model(path, compile=False)
        else:
            print(f"  ✗  Not found: {fname}  (will skip)")
            loaded[key] = None
    print()
    return loaded


# ─────────────────────────────────────────────────────────────────────────────
# INFERENCE
# ─────────────────────────────────────────────────────────────────────────────

def predict_all(img_path, models):
    inputs = {
        "cnn":      make_cnn_input(img_path),
        "transfer": make_transfer_input(img_path),
    }

    preds = {}
    for key in ["cnn", "mobilenet", "resnet", "mobilenet_ft"]:
        if models[key] is not None:
            pipe       = MODEL_PIPELINE[key]
            preds[key] = float(models[key].predict(inputs[pipe], verbose=0)[0][0])
        else:
            preds[key] = None

    available_keys = [k for k in preds if preds[k] is not None]
    weight_sum     = sum(MODEL_WEIGHTS[k] for k in available_keys)
    weighted_score = sum(MODEL_WEIGHTS[k] * preds[k]
                         for k in available_keys) / weight_sum

    votes_pneumonia = sum(1 for k in available_keys if preds[k] > BASE_THRESHOLD)
    majority_label  = ("PNEUMONIA"
                       if votes_pneumonia > len(available_keys) / 2
                       else "NORMAL")

    if weighted_score > ENSEMBLE_THRESHOLD:
        final_label = "PNEUMONIA"
    elif weighted_score < ENSEMBLE_THRESHOLD:
        final_label = "NORMAL"
    else:
        final_label = majority_label

    confidence = weighted_score if final_label == "PNEUMONIA" else 1.0 - weighted_score

    for k in preds:
        if preds[k] is None:
            preds[k] = 0.5

    ensemble = {
        "weighted_score":  weighted_score,
        "final_label":     final_label,
        "confidence":      confidence,
        "votes_pneumonia": votes_pneumonia,
        "total_models":    len(available_keys),
        "majority_label":  majority_label,
    }
    return preds, inputs, ensemble


def label(prob, threshold):
    return "PNEUMONIA" if prob > threshold else "NORMAL"


# ─────────────────────────────────────────────────────────────────────────────
# VISUALISATION
# ─────────────────────────────────────────────────────────────────────────────

COLORS = {
    "PNEUMONIA": "#E53935",
    "NORMAL":    "#43A047",
    "bg":        "#0F1117",
    "panel":     "#1A1D27",
    "border":    "#2A2D3A",
    "text":      "#E8EAF0",
    "subtext":   "#7B7F96",
}

MODEL_LABELS = {
    "cnn":          "Custom CNN",
    "mobilenet":    "MobileNetV2",
    "resnet":       "ResNet50",
    "mobilenet_ft": "MobileNet FT",
}


def _float_arr_to_bgr_uint8(float_arr_hwc):
    """Convert a (H,W,3) float32 [0,1] array to uint8 BGR for cv2 ops."""
    arr = np.clip(float_arr_hwc, 0.0, 1.0)
    return (arr * 255).astype(np.uint8)


def build_result_figure(img_path, models, preds, inputs, ensemble):
    original_bgr = cv2.imread(img_path)
    original_rgb = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB)

    # Preprocessed previews (H,W,3 uint8 for display)
    cnn_prev      = _float_arr_to_bgr_uint8(inputs["cnn"][0])
    transfer_prev = _float_arr_to_bgr_uint8(inputs["transfer"][0])

    # Source images for Grad-CAM overlay (BGR uint8)
    src_bgr = {
        "cnn":      cv2.resize(_float_arr_to_bgr_uint8(inputs["cnn"][0]),      (IMG_SIZE, IMG_SIZE)),
        "transfer": cv2.resize(_float_arr_to_bgr_uint8(inputs["transfer"][0]), (IMG_SIZE, IMG_SIZE)),
    }

    # ── Compute Grad-CAM heatmaps ─────────────────────────────────────────────
    # Build an ORDERED list of (key, overlay_rgb) only for successful heatmaps.
    # This keeps subplot indexing correct regardless of which models succeed.
    gradcam_items = []   # list of (model_key, rgb_overlay_array)

    for key in ["cnn", "mobilenet", "resnet", "mobilenet_ft"]:
        if models[key] is None:
            continue
        pipe = MODEL_PIPELINE[key]
        hm   = make_gradcam_heatmap(models[key], inputs[pipe], LAST_CONV[key])
        if hm is not None:
            overlay = overlay_gradcam(src_bgr[pipe], hm)
            gradcam_items.append((key, overlay))
        else:
            print(f"  [warn] Grad-CAM failed for {MODEL_LABELS[key]} — skipping panel")

    # ── Ensemble info ─────────────────────────────────────────────────────────
    final_label    = ensemble["final_label"]
    confidence     = ensemble["confidence"]
    weighted_score = ensemble["weighted_score"]
    verdict_col    = COLORS[final_label]

    # ── Figure layout ─────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(20, 13), facecolor=COLORS["bg"])

    outer = gridspec.GridSpec(
        4, 1, figure=fig,
        height_ratios=[0.7, 3.2, 3.2, 2.2],
        hspace=0.08, left=0.03, right=0.97, top=0.97, bottom=0.03,
    )

    # ── Row 0 : Header ────────────────────────────────────────────────────────
    ax_hdr = fig.add_subplot(outer[0])
    ax_hdr.set_facecolor(COLORS["bg"])
    ax_hdr.axis("off")
    ax_hdr.text(0.5, 0.72, "PNEUMONIA DETECTION  ·  GRAD-CAM ANALYSIS",
                ha="center", va="center", transform=ax_hdr.transAxes,
                fontsize=16, fontweight="bold", color=COLORS["text"],
                fontfamily="monospace")
    ax_hdr.text(0.5, 0.12, os.path.basename(img_path),
                ha="center", va="center", transform=ax_hdr.transAxes,
                fontsize=9, color=COLORS["subtext"], fontfamily="monospace")

    # ── Row 1 : Original + preprocessed previews ──────────────────────────────
    gs1 = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=outer[1], wspace=0.04)
    for col, (img_data, title) in enumerate([
        (original_rgb,                     "ORIGINAL  X-RAY"),
        (cv2.cvtColor(cnn_prev,      cv2.COLOR_BGR2RGB), "PREPROCESSED  ·  CNN"),
        (cv2.cvtColor(transfer_prev, cv2.COLOR_BGR2RGB), "PREPROCESSED  ·  TRANSFER"),
    ]):
        ax = fig.add_subplot(gs1[col])
        ax.imshow(img_data)
        ax.axis("off")
        ax.set_facecolor(COLORS["panel"])
        ax.set_title(title, fontsize=8, color=COLORS["subtext"],
                     fontfamily="monospace", pad=5)
        for spine in ax.spines.values():
            spine.set_edgecolor(COLORS["border"])

    # Verdict annotation
    fig.text(0.965, 0.735, f"  {final_label}  ",
             ha="right", va="top", fontsize=22, fontweight="bold",
             color=verdict_col, fontfamily="monospace",
             bbox=dict(boxstyle="round,pad=0.3", facecolor=COLORS["panel"],
                       edgecolor=verdict_col, linewidth=2))
    fig.text(0.965, 0.693, f"  confidence  {confidence*100:.1f}%  ",
             ha="right", va="top", fontsize=9,
             color=COLORS["subtext"], fontfamily="monospace")
    fig.text(0.965, 0.668,
             f"  weighted score  {weighted_score:.3f}  │  "
             f"votes {ensemble['votes_pneumonia']}/{ensemble['total_models']}  ",
             ha="right", va="top", fontsize=8,
             color=COLORS["subtext"], fontfamily="monospace")

    # ── Row 2 : Grad-CAM overlays ─────────────────────────────────────────────
    n_gc = max(len(gradcam_items), 1)
    gs2  = gridspec.GridSpecFromSubplotSpec(1, n_gc, subplot_spec=outer[2], wspace=0.04)

    # FIX: use enumerate over gradcam_items directly so subplot_idx == panel_idx
    for subplot_idx, (key, overlay_rgb) in enumerate(gradcam_items):
        prob      = preds[key]
        lbl       = label(prob, BASE_THRESHOLD)
        lbl_color = COLORS[lbl]

        ax = fig.add_subplot(gs2[subplot_idx])   # ← correct sequential index
        ax.imshow(overlay_rgb)
        ax.axis("off")
        ax.set_facecolor(COLORS["panel"])
        ax.set_title(f"GRAD-CAM  ·  {MODEL_LABELS[key]}",
                     fontsize=8, color=COLORS["subtext"],
                     fontfamily="monospace", pad=5)
        ax.text(0.5, 0.04, f"{lbl}   {prob:.3f}",
                ha="center", va="bottom", transform=ax.transAxes,
                fontsize=9, fontweight="bold", color=lbl_color,
                fontfamily="monospace",
                bbox=dict(boxstyle="round,pad=0.25",
                          facecolor=COLORS["bg"] + "CC",
                          edgecolor=lbl_color, linewidth=1.2))

    # If no heatmaps at all, show a placeholder
    if not gradcam_items:
        ax = fig.add_subplot(gs2[0])
        ax.axis("off")
        ax.set_facecolor(COLORS["panel"])
        ax.text(0.5, 0.5, "Grad-CAM unavailable",
                ha="center", va="center", transform=ax.transAxes,
                color=COLORS["subtext"], fontfamily="monospace", fontsize=10)

    # ── Row 3 : Probability bar chart ─────────────────────────────────────────
    ax_bar = fig.add_subplot(outer[3])
    ax_bar.set_facecolor(COLORS["panel"])

    keys_order = ["cnn", "mobilenet", "resnet", "mobilenet_ft"]
    x_labels   = [f"{MODEL_LABELS[k]}\n(w={MODEL_WEIGHTS[k]})" for k in keys_order]
    values     = [preds[k] for k in keys_order]
    bar_colors = [COLORS[label(v, BASE_THRESHOLD)] for v in values]

    keys_order.append("ensemble")
    x_labels.append("WEIGHTED\nENSEMBLE")
    values.append(weighted_score)
    bar_colors.append(COLORS[final_label])

    x    = np.arange(len(keys_order))
    bars = ax_bar.bar(x, values, color=bar_colors,
                      width=0.55, zorder=3, linewidth=1.5, edgecolor=COLORS["bg"])

    ax_bar.axhline(ENSEMBLE_THRESHOLD, color="#FFFFFF44", linestyle="--",
                   linewidth=1.3, zorder=2)
    ax_bar.text(len(keys_order) - 0.1, ENSEMBLE_THRESHOLD + 0.025,
                f"threshold {ENSEMBLE_THRESHOLD}", ha="right",
                color=COLORS["subtext"], fontsize=7.5, fontfamily="monospace")

    for bar, val in zip(bars, values):
        ax_bar.text(bar.get_x() + bar.get_width() / 2, val + 0.025, f"{val:.3f}",
                    ha="center", va="bottom", fontsize=9, fontweight="bold",
                    color=COLORS["text"], fontfamily="monospace")

    bars[-1].set_edgecolor(verdict_col)
    bars[-1].set_linewidth(2.5)

    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(x_labels, color=COLORS["subtext"],
                           fontfamily="monospace", fontsize=8.5)
    ax_bar.set_ylim(0, 1.18)
    ax_bar.set_ylabel("Pneumonia probability", color=COLORS["subtext"],
                      fontsize=9, fontfamily="monospace")
    ax_bar.tick_params(axis="y", colors=COLORS["subtext"])
    ax_bar.set_facecolor(COLORS["panel"])
    for spine in ax_bar.spines.values():
        spine.set_edgecolor(COLORS["border"])
    ax_bar.yaxis.grid(True, color=COLORS["border"], linewidth=0.6, zorder=0)
    ax_bar.set_axisbelow(True)

    patches = [
        mpatches.Patch(color=COLORS["PNEUMONIA"], label="Pneumonia"),
        mpatches.Patch(color=COLORS["NORMAL"],    label="Normal"),
    ]
    ax_bar.legend(handles=patches, loc="upper left",
                  facecolor=COLORS["bg"], edgecolor=COLORS["border"],
                  labelcolor=COLORS["text"], fontsize=8)

    return fig


# ─────────────────────────────────────────────────────────────────────────────
# CONSOLE SUMMARY
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(preds, ensemble):
    W = 58
    print("\n" + "═" * W)
    print(f"{'  PNEUMONIA DETECTION  RESULT':^{W}}")
    print("═" * W)
    for key, display in MODEL_LABELS.items():
        p   = preds[key]
        lbl = label(p, BASE_THRESHOLD)
        bar = "█" * int(p * 20) + "░" * (20 - int(p * 20))
        w   = MODEL_WEIGHTS[key]
        print(f"  {display:<16}  {bar}  {p:.3f}  {lbl}  (w={w})")
    print("─" * W)
    ws   = ensemble["weighted_score"]
    fl   = ensemble["final_label"]
    conf = ensemble["confidence"]
    vp   = ensemble["votes_pneumonia"]
    tot  = ensemble["total_models"]
    mv   = ensemble["majority_label"]
    print(f"  {'Weighted score':<16}  {'':20}  {ws:.3f}")
    print(f"  {'Majority vote':<16}  {'':20}  {vp}/{tot} → {mv}")
    print("═" * W)
    print(f"  FINAL VERDICT : {fl:<12}  Confidence : {conf*100:.1f}%")
    print("═" * W + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN LOOP
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 52)
    print("   PNEUMONIA DETECTOR  ·  Grad-CAM Visualizer")
    print("=" * 52)
    print(f"  Models directory : {os.path.abspath(MODELS_DIR)}")
    print(f"  Output directory : {os.path.abspath(OUTPUT_DIR)}")

    models = load_all_models(MODELS_DIR)

    base_available = sum(1 for k in ["cnn", "mobilenet", "resnet", "mobilenet_ft"]
                         if models[k] is not None)
    if base_available == 0:
        print("\n[ERROR]  No base models found.  Check MODELS_DIR path.\n")
        sys.exit(1)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("  Ready! Paste an image path and press Enter.")
    print("  Type  q  to quit.\n")

    while True:
        try:
            raw = input("Image path > ").strip().strip('"').strip("'")
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if raw.lower() in ("q", "quit", "exit", ""):
            print("Bye!")
            break

        if not os.path.isfile(raw):
            print(f"  [!] File not found: {raw}\n")
            continue

        print("  Running inference …")
        try:
            preds, inputs, ensemble = predict_all(raw, models)
        except Exception as e:
            print(f"  [ERROR] Inference failed: {e}\n")
            continue

        print_summary(preds, ensemble)

        print("  Building Grad-CAM figure …")
        try:
            fig = build_result_figure(raw, models, preds, inputs, ensemble)
            stem     = os.path.splitext(os.path.basename(raw))[0]
            out_path = os.path.join(OUTPUT_DIR, f"result_{stem}.png")
            fig.savefig(out_path, dpi=150, bbox_inches="tight",
                        facecolor=COLORS["bg"])
            plt.close(fig)
            print(f"  ✓  Saved → {os.path.abspath(out_path)}\n")
        except Exception as e:
            import traceback
            print(f"  [ERROR] Visualization failed: {e}")
            traceback.print_exc()
            print()


if __name__ == "__main__":
    main()