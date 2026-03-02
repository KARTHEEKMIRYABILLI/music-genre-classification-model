import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import json
import time
import warnings
warnings.filterwarnings("ignore")
import logging
logging.getLogger("absl").setLevel(logging.ERROR)

import numpy as np
import librosa
from datetime import datetime
from flask import Flask, request, jsonify, render_template, send_file
from flask import cli
cli.show_server_banner = lambda *args: None

import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)

import tensorflow as tf
tf.get_logger().setLevel("ERROR")

from tensorflow.keras.models import load_model
from werkzeug.utils import secure_filename
from PIL import Image

app = Flask(__name__, template_folder=".")

# ─── Configuration ────────────────────────────────────────────────────────────
UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {"wav", "mp3", "ogg", "flac", "m4a"}
MODEL_PATH = "Trained_model.h5"
HISTORY_FILE = "prediction_history.json"
MAX_HISTORY = 50

# GTZAN 10 Genre Labels (alphabetical – standard order used during training)
GENRE_LABELS = [
    "Blues", "Classical", "Country", "Disco", "Hip-Hop",
    "Jazz", "Metal", "Pop", "Reggae", "Rock"
]

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ─── Load Model ──────────────────────────────────────────────────────────────
print("[INFO] Loading model...")
model = load_model(MODEL_PATH)
print(f"[INFO] Model loaded successfully.")
print(f"[INFO] Model input shape : {model.input_shape}")
print(f"[INFO] Model output shape: {model.output_shape}")


# ─── History Helpers ──────────────────────────────────────────────────────────
def load_history():
    """Load prediction history from JSON file."""
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return []
    return []


def save_to_history(entry):
    """Append a prediction entry to history (kept to MAX_HISTORY items)."""
    history = load_history()
    history.insert(0, entry)
    history = history[:MAX_HISTORY]
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=2)


# ─── Helper Functions ─────────────────────────────────────────────────────────
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def create_mel_spectrogram(y, sr, target_height, target_width):
    """
    Create a mel-spectrogram from audio samples and resize it properly
    to (target_height, target_width) using PIL LANCZOS interpolation.
    """
    # Extract mel-spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=128, n_fft=2048, hop_length=512
    )
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    # Normalise to [0, 255] for PIL image resizing
    mel_min = mel_spec_db.min()
    mel_max = mel_spec_db.max()
    if mel_max - mel_min > 1e-8:
        mel_norm = ((mel_spec_db - mel_min) / (mel_max - mel_min) * 255).astype(np.uint8)
    else:
        mel_norm = np.zeros_like(mel_spec_db, dtype=np.uint8)

    # Use PIL for proper LANCZOS resizing (preserves spectral structure)
    img = Image.fromarray(mel_norm, mode="L")
    img_resized = img.resize((target_width, target_height), Image.LANCZOS)

    # Convert back to float [0, 1]
    result = np.array(img_resized, dtype=np.float32) / 255.0
    return result


def preprocess_audio(file_path, target_sr=22050):
    """
    Load audio and create properly resized mel-spectrograms using
    segment-based ensemble approach for robust predictions.
    Returns a batch of input arrays (one per segment).
    """
    # Load audio (mono, resampled)
    y, sr = librosa.load(file_path, sr=target_sr, mono=True)

    # Trim silence from beginning and end
    y, _ = librosa.effects.trim(y, top_db=25)

    # Normalise amplitude
    peak = np.max(np.abs(y))
    if peak > 0:
        y = y / peak

    # Get model's expected input shape (excluding batch dimension)
    expected_shape = model.input_shape[1:]  # e.g., (150, 150, 1)
    target_height = expected_shape[0]
    target_width = expected_shape[1]

    # Segment-based approach: split audio into overlapping segments
    # Each segment gets its own mel-spectrogram resized to model input size
    segment_duration = 3.0  # seconds per segment
    segment_samples = int(target_sr * segment_duration)
    hop_samples = int(segment_samples * 0.5)  # 50% overlap

    segments = []
    start = 0
    while start + segment_samples <= len(y):
        segments.append(y[start : start + segment_samples])
        start += hop_samples

    # If audio is too short, just use the whole thing (padded)
    if len(segments) == 0:
        if len(y) < segment_samples:
            y = np.pad(y, (0, segment_samples - len(y)), mode="constant")
        segments.append(y[:segment_samples])

    # Create mel-spectrogram for each segment
    batch = []
    for seg in segments:
        mel = create_mel_spectrogram(seg, sr, target_height, target_width)

        # Reshape to match model input
        if len(expected_shape) == 3:
            mel = mel.reshape(target_height, target_width, expected_shape[2])
        batch.append(mel)

    return np.array(batch)


# ─── Routes ───────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": f"Invalid file type. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"}), 400

    try:
        start_time = time.time()

        # Save uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        # Preprocess — returns a batch of segment spectrograms
        input_batch = preprocess_audio(filepath)
        segments_analyzed = len(input_batch)

        # Predict each segment
        all_predictions = model.predict(input_batch, verbose=0)

        # Average probabilities across all segments (ensemble)
        avg_predictions = np.mean(all_predictions, axis=0)

        # Get results
        predicted_index = np.argmax(avg_predictions)
        confidence = float(avg_predictions[predicted_index]) * 100

        # Build result with all genre probabilities
        genre_probs = {}
        for i, genre in enumerate(GENRE_LABELS):
            genre_probs[genre] = round(float(avg_predictions[i]) * 100, 2)

        # Sort by probability (descending)
        genre_probs = dict(sorted(genre_probs.items(), key=lambda x: x[1], reverse=True))

        processing_time = round(time.time() - start_time, 2)

        result = {
            "predicted_genre": GENRE_LABELS[predicted_index],
            "confidence": round(confidence, 2),
            "all_genres": genre_probs,
            "segments_analyzed": segments_analyzed,
            "processing_time": processing_time,
            "filename": filename,
        }

        # Save to history
        history_entry = {
            "filename": filename,
            "predicted_genre": result["predicted_genre"],
            "confidence": result["confidence"],
            "segments_analyzed": segments_analyzed,
            "processing_time": processing_time,
            "timestamp": datetime.now().isoformat(),
        }
        save_to_history(history_entry)

        # Clean up uploaded file
        os.remove(filepath)

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/model-info")
def model_info():
    """Return model metadata for display or debugging."""
    info = {
        "input_shape": str(model.input_shape),
        "output_shape": str(model.output_shape),
        "num_genres": len(GENRE_LABELS),
        "genre_labels": GENRE_LABELS,
        "total_parameters": int(model.count_params()),
        "allowed_formats": sorted(ALLOWED_EXTENSIONS),
    }
    return jsonify(info)


@app.route("/history")
def get_history():
    """Return prediction history."""
    history = load_history()
    return jsonify(history)


@app.route("/history/clear", methods=["POST"])
def clear_history():
    """Clear prediction history."""
    with open(HISTORY_FILE, "w") as f:
        json.dump([], f)
    return jsonify({"status": "cleared"})


@app.route("/download-result", methods=["POST"])
def download_result():
    """Generate a downloadable JSON report from the provided result data."""
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400

    report = {
        "report_title": "SoundSense — Genre Classification Report",
        "generated_at": datetime.now().isoformat(),
        "filename": data.get("filename", "unknown"),
        "predicted_genre": data.get("predicted_genre", ""),
        "confidence": data.get("confidence", 0),
        "all_genres": data.get("all_genres", {}),
        "segments_analyzed": data.get("segments_analyzed", 0),
        "processing_time": data.get("processing_time", 0),
    }

    report_path = os.path.join(UPLOAD_FOLDER, "report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    return send_file(report_path, as_attachment=True, download_name="soundsense_report.json")


FEEDBACK_FILE = "feedback_history.json"
FLAGGED_FILE = "flagged_predictions.json"


@app.route("/feedback", methods=["POST"])
def submit_feedback():
    """Save user feedback (good/bad) on a prediction."""
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400

    feedback_list = []
    if os.path.exists(FEEDBACK_FILE):
        try:
            with open(FEEDBACK_FILE, "r") as f:
                feedback_list = json.load(f)
        except (json.JSONDecodeError, IOError):
            feedback_list = []

    data["submitted_at"] = datetime.now().isoformat()
    feedback_list.insert(0, data)

    with open(FEEDBACK_FILE, "w") as f:
        json.dump(feedback_list, f, indent=2)

    return jsonify({"status": "received"})


@app.route("/flag", methods=["POST"])
def flag_result():
    """Flag a prediction result for review."""
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400

    # Load existing flagged items
    flagged = []
    if os.path.exists(FLAGGED_FILE):
        try:
            with open(FLAGGED_FILE, "r") as f:
                flagged = json.load(f)
        except (json.JSONDecodeError, IOError):
            flagged = []

    data["flagged_at"] = datetime.now().isoformat()
    flagged.insert(0, data)

    with open(FLAGGED_FILE, "w") as f:
        json.dump(flagged, f, indent=2)

    return jsonify({"status": "flagged"})


# ─── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Suppress Flask development server warning
    logging.getLogger("werkzeug").setLevel(logging.ERROR)
    print(" * Running on http://127.0.0.1:5000")
    print(" * Running on http://0.0.0.0:5000")
    print(" * Press CTRL+C to quit")
    app.run(debug=False, host="0.0.0.0", port=5000)
