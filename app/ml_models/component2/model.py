import tensorflow as tf
import os
import numpy as np

# Get the absolute path to where the weights and files are stored
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CLASSIFIER_PATH = os.path.join(BASE_DIR, "weights", "final_combined_classifier.keras")
AUTOENCODER_PATH = os.path.join(BASE_DIR, "weights", "final_lung_autoencoder.keras")
CENTROID_PATH = os.path.join(BASE_DIR, "files", "lung_centroid_v3.npy")
THRESHOLD_PATH = os.path.join(BASE_DIR, "files", "ood_threshold_v3.txt")
TB_CENTROID_PATH = os.path.join(BASE_DIR, "files", "tb_centroid.npy")
TB_RADIUS_PATH = os.path.join(BASE_DIR, "files", "tb_rejection_radius.txt")

_classifier = None
_autoencoder = None
_encoder = None
_lung_centroid = None
_ood_threshold = None
_tb_centroid = None
_tb_rejection_radius = None

def get_pneumonia_model():
    """Returns the MobileNetV2 classifier model, loading it lazily on first call."""
    global _classifier
    if _classifier is None:
        if not os.path.exists(CLASSIFIER_PATH):
            raise FileNotFoundError(f"Classifier weights not found at: {CLASSIFIER_PATH}")
        print(f"Loading Component 2 classifier from {CLASSIFIER_PATH}...")
        _classifier = tf.keras.models.load_model(CLASSIFIER_PATH)
    return _classifier

def get_autoencoder():
    """Returns the full autoencoder and its encoder sub-model, loading lazily on first call.
    
    The encoder outputs a 1280-dim embedding by taking the 'out_relu' layer
    (the MobileNetV2 encoder's last activation, shape 7x7x1280) and applying
    GlobalAveragePooling2D to match the precomputed centroid shape.
    """
    global _autoencoder, _encoder
    if _autoencoder is None:
        if not os.path.exists(AUTOENCODER_PATH):
            raise FileNotFoundError(f"Autoencoder weights not found at: {AUTOENCODER_PATH}")
        print(f"Loading Component 2 autoencoder from {AUTOENCODER_PATH}...")
        _autoencoder = tf.keras.models.load_model(AUTOENCODER_PATH)
        
        # The bottleneck is 'out_relu' — the last encoder layer before the decoder
        bottleneck_output = _autoencoder.get_layer("out_relu").output
        # GlobalAveragePooling2D: (None, 7, 7, 1280) -> (None, 1280)
        pooled = tf.keras.layers.GlobalAveragePooling2D()(bottleneck_output)
        
        _encoder = tf.keras.models.Model(
            inputs=_autoencoder.input,
            outputs=pooled
        )
        print(f"Encoder built: input {_encoder.input_shape} -> embedding {_encoder.output_shape}")
    return _autoencoder, _encoder

def get_ood_shield_data():
    """Returns the lung centroid vector and OOD threshold value, loading them lazily."""
    global _lung_centroid, _ood_threshold
    if _lung_centroid is None or _ood_threshold is None:
        if not os.path.exists(CENTROID_PATH):
            raise FileNotFoundError(f"Lung centroid file not found at: {CENTROID_PATH}")
        if not os.path.exists(THRESHOLD_PATH):
            raise FileNotFoundError(f"OOD threshold file not found at: {THRESHOLD_PATH}")
        
        _lung_centroid = np.load(CENTROID_PATH)
        with open(THRESHOLD_PATH, 'r') as f:
            _ood_threshold = float(f.read().strip())
            
    return _lung_centroid, _ood_threshold

def get_tb_shield_data():
    """Returns the TB centroid vector and TB rejection radius, loading them lazily."""
    global _tb_centroid, _tb_rejection_radius
    if _tb_centroid is None or _tb_rejection_radius is None:
        if not os.path.exists(TB_CENTROID_PATH):
            raise FileNotFoundError(f"TB centroid file not found at: {TB_CENTROID_PATH}")
        if not os.path.exists(TB_RADIUS_PATH):
            raise FileNotFoundError(f"TB rejection radius file not found at: {TB_RADIUS_PATH}")

        _tb_centroid = np.load(TB_CENTROID_PATH)
        with open(TB_RADIUS_PATH, "r", encoding="utf-8") as f:
            _tb_rejection_radius = float(f.read().strip())

    return _tb_centroid, _tb_rejection_radius