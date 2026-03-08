import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import gradio as gr
import pickle

# --- CONFIG ---
IMG_SIZE = 224
SEQUENCE_LENGTH = 8
MAX_RECORDING_SEC = 5 
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Paths
CSS_PATH = os.path.join(BASE_DIR, 'static', 'style.css')
HTML_PATH = os.path.join(BASE_DIR, 'templates', 'index.html')
MODEL_PATH = os.path.join(BASE_DIR, 'SignSpeak_Model_Final.keras')
LABEL_PATH = os.path.join(BASE_DIR, 'master_label_map.pkl')

# Labels
URDU_LABELS = {'aath': 'آٹھ', 'ahista': 'آہستہ', 'anywalakal': 'آنے والا کل', 'behtreen': 'بہترین', 'btana': 'بتانا', 'bukhar': 'بخار', 'bus': 'بس', 'car': 'کار', 'char': 'چار', 'chawal': 'چاول'}
ENGLISH_MAP = {'aath': 'Eight (8)', 'ahista': 'Slow', 'anywalakal': 'Tomorrow', 'behtreen': 'Perfect', 'btana': 'To Tell', 'bukhar': 'Fever', 'bus': 'Bus', 'car': 'Car', 'char': 'Four (4)', 'chawal': 'Rice'}

# --- MODEL LOADING ---
def build_model():
    base = tf.keras.applications.MobileNetV2(input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=False, weights=None)
    x = layers.GlobalAveragePooling2D()(base.output)
    feature_extractor = models.Model(inputs=base.input, outputs=x)
    model = models.Sequential([
        layers.Input(shape=(SEQUENCE_LENGTH, IMG_SIZE, IMG_SIZE, 3)),
        layers.TimeDistributed(feature_extractor),
        layers.Bidirectional(layers.LSTM(128)),
        layers.BatchNormalization(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(10, activation='softmax')
    ])
    return model

model = build_model()
model.load_weights(MODEL_PATH)
with open(LABEL_PATH, 'rb') as f:
    inv_label_map = {int(v): k for k, v in pickle.load(f).items()}

# --- IMPROVED PREDICTION LOGIC ---
def predict_sign(video_path):
    if not video_path: return "No Video Found"
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps < 5 or np.isnan(fps): fps = 25 # Standard for Webcam
    
    # Logic: Read only first 5 seconds
    max_frames = int(fps * MAX_RECORDING_SEC)
    raw_frames = []
    
    while len(raw_frames) < max_frames:
        ret, frame = cap.read()
        if not ret: break
        # Quality fix: Convert to RGB immediately
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        raw_frames.append(frame)
    cap.release()

    if len(raw_frames) < SEQUENCE_LENGTH: 
        return "Video is too short for prediction."
    
    # Temporal Sampling: Select 8 frames across the 5s window
    indices = np.linspace(0, len(raw_frames) - 1, SEQUENCE_LENGTH, dtype=int)
    
    processed_frames = []
    for i in indices:
        f = raw_frames[i]
        f = cv2.resize(f, (IMG_SIZE, IMG_SIZE))
        # Zero-centering normalization
        f = (f.astype(np.float32) / 127.5) - 1.0 
        processed_frames.append(f)
        
    input_data = np.expand_dims(np.array(processed_frames, dtype=np.float32), axis=0)
    
    # Prediction with Softmax
    preds = model.predict(input_data, verbose=0)
    idx_p = int(np.argmax(preds))
    conf = np.max(preds) * 100
    
    label_eng = inv_label_map.get(idx_p, "Unknown")
    urdu = URDU_LABELS.get(label_eng, label_eng)
    meaning = ENGLISH_MAP.get(label_eng, label_eng)
    
    return f"Urdu: {urdu}\nEnglish: {meaning}\nConfidence: {conf:.2f}%"

# --- INTERFACE ---
with open(CSS_PATH, "r") as f: css_content = f.read()
with open(HTML_PATH, "r") as f: html_content = f.read()


title_str = "SignSpeak AI \U0001F91F"

with gr.Blocks(css=css_content, title=title_str) as demo:
    gr.HTML(html_content)
    with gr.Row():
        with gr.Column(scale=1, elem_classes="white-card"):
            gr.HTML("<br>")
            predict_btn = gr.Button("PREDICT", elem_id="predict-btn")
            reset_btn = gr.Button("RESET", elem_id="reset-btn")
        
        with gr.Column(scale=3, elem_classes="white-card"):
            gr.HTML("<span class='section-header'>VIDEO INPUT (Max 5 Sec)</span>")
            vid_input = gr.Video(label="", show_label=False, sources=["upload", "webcam"], height=350)
            
        with gr.Column(scale=1, elem_classes="white-card"):
            gr.HTML("<span class='section-header'>PREDICTION RESULT</span>")
            # Result box without 'Textbox' header
            text_out = gr.Textbox(label="", show_label=False, lines=5, elem_classes="output-text", placeholder="Result...")

    predict_btn.click(predict_sign, vid_input, text_out)
    reset_btn.click(lambda: (None, ""), None, [vid_input, text_out])

if __name__ == "__main__":
    demo.launch(share=True)