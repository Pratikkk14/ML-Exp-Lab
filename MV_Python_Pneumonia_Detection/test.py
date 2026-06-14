"""
=============================================================
  Pneumonia Detection — Single Image Tester with Grad-CAM
=============================================================

SETUP
-----
Place your .h5 model files in ONE folder and set MODELS_DIR below.

Expected filenames  (exact match):
  cnn_model.h5
  MobileNet_model.h5
  model_transferLearn_RestNet50.h5
  FineTuneMobileNet.h5
  meta_model.h5   ← optional; falls back to average if missing

USAGE
-----
  python pneumonia_gradcam_tester.py

Then just paste / drag-drop an image path when prompted.
Type  q  to quit.

Each run saves a result PNG next to this script.
=============================================================
"""

# ── USER CONFIG ──────────────────────────────────────────────────────────────

MODELS_DIR = "."          # ← folder that contains your .h5 files

OUTPUT_DIR = "."          # ← where result PNGs are saved

# Classification thresholds
BASE_THRESHOLD = 0.6      # individual model (matches notebook)
META_THRESHOLD = 0.5      # meta / ensemble model

GRADCAM_ALPHA  = 0.45     # heatmap blend strength  (0 = none, 1 = full)

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
# PREPROCESSING  (exact pipeline from notebook)
# ─────────────────────────────────────────────────────────────────────────────

def _clean_mask(mask):
    k = np.ones((5, 5), np.uint8)
    return cv2.morphologyEx(cv2.morphologyEx(mask, cv2.MORPH_OPEN, k), cv2.MORPH_CLOSE, k)


def _best_contour(contours):
    big = [c for c in contours if cv2.contourArea(c) > 500]
    return max(big, key=cv2.contourArea) if big else None


def segment_cnn(img_bgr):
    """Segmentation used for the custom CNN branch."""
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
    """Segmentation used for transfer-learning models (segment_for_transfer)."""
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
    """Slight shift augmentation used in CNN pipeline."""
    M = np.float32([[1, 0, 5], [0, 1, 5]])
    return cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))


def make_cnn_input(img_path):
    """Load → blur → segment_cnn → register → normalise → (1,224,224,3)."""
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
    """Load → segment_transfer → normalise → (1,224,224,3)."""
    img = cv2.imread(img_path)
    roi = segment_transfer(img)
    out = roi.astype(np.float32) / 255.0
    if out.ndim == 2:
        out = np.stack([out] * 3, axis=-1)
    return np.expand_dims(out, 0)


# ─────────────────────────────────────────────────────────────────────────────
# GRAD-CAM  (from notebook implementation)
# ─────────────────────────────────────────────────────────────────────────────

# Last conv layer names — matches each model architecture
LAST_CONV = {
    "cnn":     "conv2d_2",          # 3rd Conv2D in custom CNN
    "mobilenet":    "Conv_1",        # last conv block in MobileNetV2
    "resnet":       "conv5_block3_out",  # last residual output in ResNet50
    "mobilenet_ft": "Conv_1",        # same backbone as MobileNet
}


def _find_last_conv(model):
    """Auto-detect last Conv2D layer as a safe fallback."""
    for layer in reversed(model.layers):
        if isinstance(layer, (tf.keras.layers.Conv2D,
                               tf.keras.layers.DepthwiseConv2D)):
            return layer.name
    return None


def make_gradcam_heatmap(model, img_array, last_conv_name):
    """
    Compute Grad-CAM heatmap (float32, shape HxW, range [0,1]).
    Mirrors notebook's make_gradcam_heatmap function exactly.
    """
    try:
        model.get_layer(last_conv_name)
    except ValueError:
        fallback = _find_last_conv(model)
        if fallback is None:
            return None
        last_conv_name = fallback

    grad_model = Model(
        inputs  = model.inputs,
        outputs = [model.get_layer(last_conv_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        inp = tf.cast(img_array, tf.float32)
        conv_outputs, predictions = grad_model(inp)
        class_channel = predictions[:, 0]          # binary sigmoid output

    grads        = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_out     = conv_outputs[0]
    heatmap      = conv_out @ pooled_grads[..., tf.newaxis]
    heatmap      = tf.squeeze(heatmap)
    heatmap      = tf.maximum(heatmap, 0)
    max_val      = tf.reduce_max(heatmap)
    heatmap      = heatmap / (max_val + 1e-8)
    return heatmap.numpy()


def overlay_gradcam(img_bgr, heatmap, alpha=GRADCAM_ALPHA):
    """
    Resize heatmap, apply JET colormap, blend onto image.
    Mirrors notebook's overlay_gradcam function exactly.
    Returns RGB uint8.
    """
    h, w = img_bgr.shape[:2]
    hm   = cv2.resize(heatmap, (w, h))
    hm   = np.uint8(255 * hm)
    hm   = cv2.applyColorMap(hm, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img_bgr, 1 - alpha, hm, alpha, 0)
    return cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)


# ─────────────────────────────────────────────────────────────────────────────
# MODEL LOADING
# ─────────────────────────────────────────────────────────────────────────────

MODEL_FILES = {
    "cnn":          "cnn_model.h5",
    "mobilenet":    "MobileNet_model.h5",
    "resnet":       "model_transferLearn_RestNet50.h5",
    "mobilenet_ft": "FineTuneMobileNet.h5",
    "meta":         "meta_model.h5",
}

# Which preprocessing each base model uses
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
    """Run all base models + meta model. Returns dict of float predictions."""
    inputs = {
        "cnn":  make_cnn_input(img_path),
        "transfer": make_transfer_input(img_path),
    }

    preds = {}
    for key in ["cnn", "mobilenet", "resnet", "mobilenet_ft"]:
        if models[key] is not None:
            pipe = MODEL_PIPELINE[key]
            preds[key] = float(models[key].predict(inputs[pipe], verbose=0)[0][0])
        else:
            preds[key] = 0.5   # neutral placeholder

    # Meta model
    if models["meta"] is not None:
        meta_in  = np.array([[preds["cnn"], preds["mobilenet"],
                               preds["resnet"], preds["mobilenet_ft"]]])
        preds["meta"] = float(models["meta"].predict(meta_in, verbose=0)[0][0])
    else:
        preds["meta"] = float(np.mean([preds["cnn"], preds["mobilenet"],
                                        preds["resnet"], preds["mobilenet_ft"]]))

    return preds, inputs


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


def build_result_figure(img_path, models, preds, inputs):
    original_bgr = cv2.imread(img_path)
    original_rgb = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB)
    orig_disp    = cv2.resize(original_bgr, (IMG_SIZE, IMG_SIZE))

    # Preprocessed previews
    cnn_prev      = (inputs["cnn"][0]      * 255).astype(np.uint8)
    transfer_prev = (inputs["transfer"][0] * 255).astype(np.uint8)

    # Grad-CAM for each base model
    gradcam_overlays = {}
    for key in ["cnn", "mobilenet", "resnet", "mobilenet_ft"]:
        if models[key] is None:
            continue
        pipe   = MODEL_PIPELINE[key]
        inp    = inputs[pipe]
        hm     = make_gradcam_heatmap(models[key], inp, LAST_CONV[key])
        if hm is not None:
            src_bgr = cv2.resize(
                inputs["cnn"][0] if pipe == "cnn" else inputs["transfer"][0],
                (IMG_SIZE, IMG_SIZE)
            )
            src_bgr = (src_bgr * 255).astype(np.uint8)
            gradcam_overlays[key] = overlay_gradcam(src_bgr, hm)

    # ── Final verdict ─────────────────────────────────────────────────────────
    meta_prob   = preds["meta"]
    final_label = label(meta_prob, META_THRESHOLD)
    confidence  = meta_prob if final_label == "PNEUMONIA" else 1 - meta_prob
    verdict_col = COLORS[final_label]

    # ── Figure ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(20, 13), facecolor=COLORS["bg"])

    # Layout: header row | images row | gradcam row | bar row
    outer = gridspec.GridSpec(4, 1, figure=fig,
                              height_ratios=[0.7, 3.2, 3.2, 2.2],
                              hspace=0.08, left=0.03, right=0.97,
                              top=0.97, bottom=0.03)

    # ── Row 0 : Header ────────────────────────────────────────────────────────
    ax_hdr = fig.add_subplot(outer[0])
    ax_hdr.set_facecolor(COLORS["bg"])
    ax_hdr.axis("off")
    ax_hdr.text(0.5, 0.72, "PNEUMONIA DETECTION  ·  GRAD-CAM ANALYSIS",
                ha="center", va="center", transform=ax_hdr.transAxes,
                fontsize=16, fontweight="bold", color=COLORS["text"],
                fontfamily="monospace", letterspacing=2)
    ax_hdr.text(0.5, 0.12, os.path.basename(img_path),
                ha="center", va="center", transform=ax_hdr.transAxes,
                fontsize=9, color=COLORS["subtext"], fontfamily="monospace")

    # ── Row 1 : Originals + preprocessed ─────────────────────────────────────
    gs1 = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=outer[1],
                                           wspace=0.04)
    panels_r1 = [
        (original_rgb,  "ORIGINAL  X-RAY",             None),
        (cnn_prev,      "PREPROCESSED  ·  CNN",         None),
        (transfer_prev, "PREPROCESSED  ·  TRANSFER",    None),
    ]
    for col, (img_data, title, _) in enumerate(panels_r1):
        ax = fig.add_subplot(gs1[col])
        ax.imshow(img_data)
        ax.axis("off")
        ax.set_facecolor(COLORS["panel"])
        ax.set_title(title, fontsize=8, color=COLORS["subtext"],
                     fontfamily="monospace", pad=5)
        for spine in ax.spines.values():
            spine.set_edgecolor(COLORS["border"])

    # Verdict box in col 2 header area (draw as annotation)
    fig.text(0.965, 0.735,
             f"  {final_label}  ",
             ha="right", va="top",
             fontsize=22, fontweight="bold",
             color=verdict_col, fontfamily="monospace",
             bbox=dict(boxstyle="round,pad=0.3",
                       facecolor=COLORS["panel"],
                       edgecolor=verdict_col, linewidth=2))
    fig.text(0.965, 0.690,
             f"  confidence  {confidence*100:.1f}%  ",
             ha="right", va="top", fontsize=9,
             color=COLORS["subtext"], fontfamily="monospace")

    # ── Row 2 : Grad-CAM overlays ─────────────────────────────────────────────
    n_gc = len(gradcam_overlays)
    gs2  = gridspec.GridSpecFromSubplotSpec(1, max(n_gc, 1), subplot_spec=outer[2],
                                            wspace=0.04)
    for idx, key in enumerate(["cnn", "mobilenet", "resnet", "mobilenet_ft"]):
        if key not in gradcam_overlays:
            continue
        prob       = preds[key]
        lbl        = label(prob, BASE_THRESHOLD)
        lbl_color  = COLORS[lbl]
        ax = fig.add_subplot(gs2[idx])
        ax.imshow(gradcam_overlays[key])
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

    # ── Row 3 : Probability bar chart ─────────────────────────────────────────
    ax_bar = fig.add_subplot(outer[3])
    ax_bar.set_facecolor(COLORS["panel"])

    keys_order  = ["cnn", "mobilenet", "resnet", "mobilenet_ft", "meta"]
    x_labels    = [MODEL_LABELS.get(k, "META") for k in keys_order]
    x_labels[-1] = "META  (ensemble)"
    values      = [preds[k] for k in keys_order]
    bar_colors  = [COLORS[label(v, META_THRESHOLD if k == "meta" else BASE_THRESHOLD)]
                   for k, v in zip(keys_order, values)]

    x = np.arange(len(keys_order))
    bars = ax_bar.bar(x, values, color=bar_colors,
                      width=0.55, zorder=3,
                      linewidth=1.5, edgecolor=COLORS["bg"])

    ax_bar.axhline(META_THRESHOLD, color="#FFFFFF33", linestyle="--",
                   linewidth=1.2, zorder=2)
    ax_bar.text(len(keys_order) - 0.1, META_THRESHOLD + 0.02,
                f"threshold {META_THRESHOLD}", ha="right",
                color=COLORS["subtext"], fontsize=7.5, fontfamily="monospace")

    for bar, val in zip(bars, values):
        ax_bar.text(bar.get_x() + bar.get_width() / 2,
                    val + 0.025, f"{val:.3f}",
                    ha="center", va="bottom",
                    fontsize=9, fontweight="bold",
                    color=COLORS["text"], fontfamily="monospace")

    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(x_labels, color=COLORS["subtext"],
                           fontfamily="monospace", fontsize=9)
    ax_bar.set_ylim(0, 1.12)
    ax_bar.set_ylabel("Pneumonia probability",
                      color=COLORS["subtext"], fontsize=9,
                      fontfamily="monospace")
    ax_bar.tick_params(axis="y", colors=COLORS["subtext"])
    ax_bar.set_facecolor(COLORS["panel"])
    for spine in ax_bar.spines.values():
        spine.set_edgecolor(COLORS["border"])
    ax_bar.yaxis.grid(True, color=COLORS["border"], linewidth=0.6, zorder=0)
    ax_bar.set_axisbelow(True)

    # Legend
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

def print_summary(preds):
    meta_prob   = preds["meta"]
    final_label = label(meta_prob, META_THRESHOLD)
    confidence  = meta_prob if final_label == "PNEUMONIA" else 1 - meta_prob

    W = 52
    print("\n" + "═" * W)
    print(f"{'  RESULT':^{W}}")
    print("═" * W)
    for key, display in MODEL_LABELS.items():
        p   = preds[key]
        lbl = label(p, BASE_THRESHOLD)
        bar = "█" * int(p * 20) + "░" * (20 - int(p * 20))
        print(f"  {display:<16}  {bar}  {p:.3f}  {lbl}")
    print("─" * W)
    print(f"  {'META (ensemble)':<16}  {'':20}  {meta_prob:.3f}  {final_label}")
    print("═" * W)
    print(f"  FINAL : {final_label}   |   Confidence : {confidence*100:.1f}%")
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

        print(f"  Running inference …")
        try:
            preds, inputs = predict_all(raw, models)
        except Exception as e:
            print(f"  [ERROR] Inference failed: {e}\n")
            continue

        print_summary(preds)

        print("  Building Grad-CAM figure …")
        try:
            fig = build_result_figure(raw, models, preds, inputs)
            stem     = os.path.splitext(os.path.basename(raw))[0]
            out_path = os.path.join(OUTPUT_DIR, f"result_{stem}.png")
            fig.savefig(out_path, dpi=150, bbox_inches="tight",
                        facecolor=COLORS["bg"])
            plt.close(fig)
            print(f"  ✓  Saved → {os.path.abspath(out_path)}\n")
        except Exception as e:
            print(f"  [ERROR] Visualization failed: {e}\n")


if __name__ == "__main__":
    main()