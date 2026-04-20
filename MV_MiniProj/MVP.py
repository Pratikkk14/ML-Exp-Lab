# =========================================================
# DROWSINESS DETECTION - STANDALONE INFERENCE SCRIPT (FIXED)
# =========================================================

import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from scipy.spatial import distance as dist

# NEW MEDIAPIPE (TASKS API)
from mediapipe.tasks.python import vision
from mediapipe.tasks import python as mp_tasks

# ---------------- CONFIG ----------------
IMG_SIZE = 145
MODEL_PATH = "D:\\Projects\\MV_Python\\MV_MiniProj\\drowsiness_detection_v2.keras"
IMAGE_PATH = "D:\\Projects\\MV_Python\\MV_MiniProj\\active1.jpg"
MODEL_TASK_PATH = "D:\\Projects\\MV_Python\\MV_MiniProj\\face_landmarker.task"    

# ---------------- LOAD MODEL ----------------
model = load_model(MODEL_PATH)
print("✅ Model loaded")

# ---------------- LOAD MEDIAPIPE MODEL ----------------
base_options = mp_tasks.BaseOptions(model_asset_path=MODEL_TASK_PATH)

options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    num_faces=1
)

face_landmarker = vision.FaceLandmarker.create_from_options(options)

# ---------------- EAR LANDMARKS ----------------
LEFT_EYE  = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

# ---------------- PREPROCESS ----------------
def preprocess_image(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(2.0, (8, 8))
    l = clahe.apply(l)
    img = cv2.merge([l, a, b])
    img = cv2.cvtColor(img, cv2.COLOR_LAB2BGR)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    return img

# ---------------- EAR ----------------
def eye_aspect_ratio(pts):
    P1, P2, P3, P4, P5, P6 = pts
    A = dist.euclidean(P2, P6)
    B = dist.euclidean(P3, P5)
    C = dist.euclidean(P1, P4)
    return (A + B) / (2.0 * C) if C > 0 else 0

def extract_ear(points):
    left = [points[i] for i in LEFT_EYE]
    right = [points[i] for i in RIGHT_EYE]

    left_ear = eye_aspect_ratio(left)
    right_ear = eye_aspect_ratio(right)

    return (left_ear + right_ear) / 2

# ---------------- MAIN PREDICT ----------------
def predict(image_path):

    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(image_path)

    img = preprocess_image(img)

    # FACE DETECTION
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        print("❌ No face detected")
        return

    # largest face
    (x, y, w, h) = sorted(faces, key=lambda b: b[2]*b[3], reverse=True)[0]
    face = img[y:y+h, x:x+w]

    # ---------------- MEDIAPIPE TASK ----------------
    rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

    mp_image = mp_tasks.Image(
        image_format=mp_tasks.ImageFormat.SRGB,
        data=rgb
    )

    result = face_landmarker.detect(mp_image)

    ear = 0.0

    if result.face_landmarks:
        landmarks = result.face_landmarks[0]

        points = []
        for lm in landmarks:
            px = int(lm.x * w)
            py = int(lm.y * h)
            points.append((px, py))

        ear = extract_ear(points)

    # ---------------- MODEL INPUT ----------------
    face_resized = cv2.resize(face, (IMG_SIZE, IMG_SIZE)) / 255.0

    img_input = face_resized[np.newaxis, ...]
    ear_input = np.array([[ear, ear, ear]])

    prob = model.predict(
        {"image_input": img_input, "ear_input": ear_input},
        verbose=0
    )[0][0]

    label = "Active" if prob >= 0.5 else "Fatigue"

    # ---------------- DRAW OUTPUT ----------------
    color = (0,255,0) if label=="Active" else (0,0,255)

    cv2.rectangle(img, (x,y), (x+w,y+h), color, 2)

    text = f"{label} ({prob*100:.1f}%) EAR:{ear:.3f}"
    cv2.putText(img, text, (x, y-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.title("Drowsiness Prediction")
    plt.show()

    print("\n--- RESULT ---")
    print("Prediction :", label)
    print("Confidence :", prob)
    print("EAR        :", ear)


# ---------------- RUN ----------------
if __name__ == "__main__":
    predict(IMAGE_PATH)