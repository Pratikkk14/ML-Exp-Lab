import time
from collections import deque
from pathlib import Path
from threading import Lock

import av
import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import VideoProcessorBase, WebRtcMode, webrtc_streamer
from tensorflow.keras.models import load_model


st.set_page_config(
	page_title="Smart Connect",
	page_icon="📹",
	layout="wide",
	initial_sidebar_state="collapsed",
)


st.markdown(
	"""
	<style>
	@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

	:root {
		--bg-start: #0f1419;
		--bg-end: #1a1f2e;
		--accent: #00d4b8;
		--danger: #ff3b5c;
		--text: #f0f0f0;
		--muted: #a0a0a0;
		--card: rgba(255, 255, 255, 0.05);
		--border: rgba(255, 255, 255, 0.1);
	}

	html, body, [class*="css"] {
		font-family: 'Inter', sans-serif;
	}

	.stApp {
		background: linear-gradient(135deg, var(--bg-start), var(--bg-end));
		color: var(--text);
	}

	.header-container {
		background: var(--card);
		border: 1px solid var(--border);
		border-radius: 14px;
		padding: 1.5rem;
		margin-bottom: 1.5rem;
		backdrop-filter: blur(10px);
	}

	.header-container h1 {
		font-size: 2.2rem;
		font-weight: 700;
		margin: 0;
		color: var(--text);
	}

	.header-container p {
		margin: 0.5rem 0 0 0;
		color: var(--muted);
		font-size: 0.95rem;
	}

	.alert-drowsy {
		border: 1px solid rgba(255, 59, 92, 0.6);
		background: rgba(255, 59, 92, 0.12);
		color: #ff8fa3;
		border-radius: 12px;
		padding: 1rem;
		font-weight: 600;
		margin-bottom: 1rem;
		animation: pulse 1.5s infinite;
	}

	@keyframes pulse {
		0%, 100% { opacity: 1; }
		50% { opacity: 0.8; }
	}

	.stat-row {
		display: grid;
		grid-template-columns: repeat(2, 1fr);
		gap: 1rem;
		margin-top: 1rem;
	}

	.stat-item {
		background: rgba(255, 255, 255, 0.03);
		border: 1px solid var(--border);
		border-radius: 10px;
		padding: 0.9rem;
	}

	.stat-label {
		font-size: 0.85rem;
		color: var(--muted);
		margin-bottom: 0.3rem;
	}

	.stat-value {
		font-size: 1.3rem;
		font-weight: 700;
		color: var(--accent);
	}
	</style>
	""",
	unsafe_allow_html=True,
)


@st.cache_resource
def load_model_safe(model_path: str):
	try:
		return load_model(model_path)
	except Exception as e:
		st.error(f"Failed to load model: {e}")
		st.stop()


def preprocess_frame(frame_bgr: np.ndarray, model_input_shape):
	_, h, w, c = model_input_shape
	img = cv2.resize(frame_bgr, (w, h))
	if c == 1:
		img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		img = np.expand_dims(img, axis=-1)

	img = img.astype("float32") / 255.0
	x = np.expand_dims(img, axis=0)
	return x


class DrowsinessProcessor(VideoProcessorBase):
	def __init__(self, model, threshold: float, sample_interval_sec: float):
		self.model = model
		self.threshold = threshold
		self.sample_interval_sec = sample_interval_sec
		self.last_sample_time = 0.0
		self.last_score = None
		self.last_label = "Initializing..."
		self.last_alert = ""
		self.sample_count = 0
		self.recent_scores = deque(maxlen=20)
		self._lock = Lock()

	def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
		frame_bgr = frame.to_ndarray(format="bgr24")
		now = time.time()

		if now - self.last_sample_time >= self.sample_interval_sec:
			try:
				x = preprocess_frame(frame_bgr, self.model.input_shape)
				pred = self.model.predict(x, verbose=0)

				if pred.shape[-1] == 1:
					drowsy_score = float(pred[0][0])
				else:
					drowsy_score = float(pred[0][1])

				label = "Drowsy" if drowsy_score >= self.threshold else "Awake"

				with self._lock:
					self.last_sample_time = now
					self.last_score = drowsy_score
					self.last_label = label
					self.sample_count += 1
					self.recent_scores.append(drowsy_score)

					if label == "Drowsy":
						self.last_alert = f"🚨 WAKEUP! | Score: {drowsy_score:.3f} | {time.strftime('%H:%M:%S')}"
			except Exception:
				pass

		with self._lock:
			label = self.last_label
			score = self.last_score

		if score is not None:
			color = (0, 0, 255) if label == "Drowsy" else (0, 255, 0)
			thickness = 2 if label == "Drowsy" else 1
			msg = f"{label} ({score:.2f})"
			cv2.putText(
				frame_bgr,
				msg,
				(15, 35),
				cv2.FONT_HERSHEY_SIMPLEX,
				1.0,
				color,
				thickness,
				cv2.LINE_AA,
			)

		return av.VideoFrame.from_ndarray(frame_bgr, format="bgr24")

	def get_stats(self):
		with self._lock:
			avg_score = float(np.mean(self.recent_scores)) if self.recent_scores else None
			return {
				"sample_count": self.sample_count,
				"last_score": self.last_score,
				"avg_score": avg_score,
				"last_label": self.last_label,
				"last_alert": self.last_alert,
			}


project_root = Path(__file__).resolve().parent
model_path = project_root / "my_model.h5"

if not model_path.exists():
	st.error(f"❌ Model not found: {model_path}")
	st.stop()

model = load_model_safe(str(model_path))

st.markdown(
	"""
	<div class="header-container">
	  <h1>📹 Smart Connect</h1>
	  <p>Real-time drowsiness detection using USB camera and TensorFlow model</p>
	</div>
	""",
	unsafe_allow_html=True,
)

threshold = 0.5
sample_interval = 3.0 / 5.0

media_constraints = {
	"video": {
		"width": {"ideal": 640},
		"height": {"ideal": 480},
		"frameRate": {"ideal": 24},
	},
	"audio": False,
}

col_cam, col_info = st.columns([1.8, 1], gap="medium")

with col_cam:
	st.subheader("Camera Feed")
	st.caption("Allow camera access. Model runs continuously.")

	ctx = webrtc_streamer(
		key="smart-connect-stream",
		mode=WebRtcMode.SENDRECV,
		rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
		media_stream_constraints=media_constraints,
		async_processing=True,
		video_processor_factory=lambda: DrowsinessProcessor(
			model=model,
			threshold=threshold,
			sample_interval_sec=sample_interval,
		),
	)

with col_info:
	st.subheader("Status")

	alert_slot = st.empty()
	stat1 = st.empty()
	stat2 = st.empty()
	stat3 = st.empty()
	stat4 = st.empty()

	if ctx.state.playing:
		placeholder = st.empty()
		while ctx.state.playing:
			if ctx.video_processor:
				stats = ctx.video_processor.get_stats()

				if stats["last_alert"]:
					with alert_slot.container():
						st.markdown(
							f"<div class='alert-drowsy'>{stats['last_alert']}</div>",
							unsafe_allow_html=True,
						)

				stat1.metric("State", stats["last_label"])
				stat2.metric("Samples", stats["sample_count"])

				if stats["last_score"] is not None:
					stat3.metric("Score", f"{stats['last_score']:.3f}")
				else:
					stat3.metric("Score", "-")

				if stats["avg_score"] is not None:
					stat4.metric("Avg (recent)", f"{stats['avg_score']:.3f}")
				else:
					stat4.metric("Avg (recent)", "-")

			time.sleep(0.3)
	else:
		st.info("👆 Click START above to begin monitoring")

st.markdown("---")
st.caption(
	"Model: my_model.h5 | Inference: 5 samples/3sec | Run: `streamlit run MV_Python_MINIProj.py`"
)
