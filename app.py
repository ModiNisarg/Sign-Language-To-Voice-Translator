# app.py
import os
import threading
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration, WebRtcMode
import av
import cv2
import numpy as np
from keras.models import load_model
from cvzone.HandTrackingModule import HandDetector
from string import ascii_uppercase
import pyttsx3

# ----------------------------
# Config
# ----------------------------
MODEL_PATH = "trainedmodel.h5"
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# ----------------------------
# Load model once
# ----------------------------
@st.cache_resource
def load_model_once():
    return load_model(MODEL_PATH)

model = load_model_once()
hd = HandDetector(maxHands=1)

# ----------------------------
# Predictor wrapper
# ----------------------------
class SignPredictor:
    def __init__(self, model_ref):
        self.model = model_ref
        self.sentence = ""

    def predict(self, img: np.ndarray) -> str:
        # model expects (1, 400, 400, 3)
        try:
            resized = cv2.resize(img, (400, 400))
        except Exception:
            return ""
        arr = np.asarray(resized).reshape(1, 400, 400, 3)
        preds = self.model.predict(arr)[0]
        idx = int(np.argmax(preds))
        letters = list(ascii_uppercase)
        return letters[idx] if idx < len(letters) else "?"

# global predictor instance
predictor = SignPredictor(model)

# ----------------------------
# Video transformer
# ----------------------------
class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.predictor = predictor

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)

        hands = hd.findHands(img, draw=False, flipType=True)
        if hands:
            hand = hands[0]
            x, y, w, h = hand["bbox"]
            x1, y1 = max(0, x - 20), max(0, y - 20)
            x2, y2 = min(img.shape[1], x + w + 20), min(img.shape[0], y + h + 20)
            cropped = img[y1:y2, x1:x2]
            if cropped.size > 0:
                pred = self.predictor.predict(cropped)
                if pred:
                    self.predictor.sentence += pred
                cv2.putText(img, f"Pred: {pred}", (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

        cv2.putText(img, f"Sentence: {self.predictor.sentence}", (10, img.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# ----------------------------
# TTS
# ----------------------------
def speak_server(text: str):
    try:
        engine = pyttsx3.init()
        engine.setProperty("rate", 150)
        engine.say(text)
        engine.runAndWait()
    except Exception as e:
        print("TTS error:", e)

# ----------------------------
# Streamlit UI
# ----------------------------
def main():
    st.set_page_config(page_title="Sign-to-Text", layout="wide")
    st.title("🖐️ Sign Language → Text")

    if not os.path.exists(MODEL_PATH):
        st.error(f"Model not found: {MODEL_PATH}")
        st.stop()

  webrtc_ctx = webrtc_streamer(
    key="sign-to-text",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"video": True, "audio": False},
    video_transformer_factory=VideoTransformer,
    # async_transform=True,   # disable for testing
)


    st.sidebar.title("Controls")
    sentence_box = st.empty()
    speak_btn = st.sidebar.button("Speak sentence")
    clear_btn = st.sidebar.button("Clear sentence")

    if webrtc_ctx and webrtc_ctx.video_transformer:
        tr = webrtc_ctx.video_transformer.predictor
        sentence_box.markdown(f"**Sentence:** `{tr.sentence}`")

        if speak_btn:
            threading.Thread(target=speak_server, args=(tr.sentence,)).start()
        if clear_btn:
            tr.sentence = ""
    else:
        sentence_box.markdown("**Sentence:** `(not started yet)`")

if __name__ == "__main__":
    main()