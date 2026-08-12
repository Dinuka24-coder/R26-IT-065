"""End-to-end validation for Component 2 OOD + TB shields."""

import glob
import os
import sys

import cv2
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from app.ml_models.component2.inference import run_pneumonia_inference, InvalidXRayError, _apply_clahe
from app.ml_models.component2.model import get_autoencoder, get_ood_shield_data, get_tb_shield_data

TB_DIR = r"D:\Datasets\TB\TB\TB_Chest_Radiography_Database\Tuberculosis"
PEDIATRIC_TEST_DIR = r"D:\Datasets\Pneumonia\archive\chest_xray\test"
RSNA_DIR = r"D:\Datasets\Pneumonia\rsna_1000_samples"


def tb_shield_would_reject(feat_vector):
    lung_centroid, ood_threshold = get_ood_shield_data()
    tb_centroid, tb_radius = get_tb_shield_data()

    distance = float(np.linalg.norm(feat_vector - lung_centroid))
    tb_distance = float(np.linalg.norm(feat_vector - tb_centroid))

    if distance > ood_threshold:
        return "ood_reject", distance, tb_distance
    if tb_distance <= tb_radius and distance > tb_distance:
        return "tb_reject", distance, tb_distance
    return "pass", distance, tb_distance


def embed_image(path):
    img = cv2.imread(path)
    if img is None:
        return None

    img_resized = cv2.resize(img, (224, 224))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_clahe = _apply_clahe(img_rgb)
    img_ae = np.expand_dims(img_clahe, axis=0).astype("float32") / 255.0

    _, encoder = get_autoencoder()
    return encoder.predict(img_ae, verbose=0).flatten()


def evaluate_folder(name, folder, limit=None):
    paths = sorted(glob.glob(os.path.join(folder, "*.png")))
    paths += sorted(glob.glob(os.path.join(folder, "*.jpg")))
    paths += sorted(glob.glob(os.path.join(folder, "*.jpeg")))
    if limit:
        paths = paths[:limit]

    passed = failed = 0
    for path in paths:
        img = cv2.imread(path)
        if img is None:
            continue
        try:
            run_pneumonia_inference(img)
            passed += 1
        except InvalidXRayError:
            failed += 1

    total = passed + failed
    print(f"{name:28} {total:5} {passed:5} {failed:5} {100*passed/total if total else 0:6.1f}% {100*failed/total if total else 0:6.1f}%")
    return passed, failed


def main():
    print("=" * 72)
    print("Component 2 Shield Validation")
    print("=" * 72)

    _, tb_radius = get_tb_shield_data()
    print(f"TB rejection radius: {tb_radius:.6f}\n")
    print(f"{'Category':28} {'Total':>5} {'Pass':>5} {'Fail':>5} {'Pass%':>7} {'Fail%':>7}")
    print("-" * 72)

    evaluate_folder("Pediatric NORMAL", os.path.join(PEDIATRIC_TEST_DIR, "NORMAL"))
    evaluate_folder("Pediatric PNEUMONIA", os.path.join(PEDIATRIC_TEST_DIR, "PNEUMONIA"))
    evaluate_folder("RSNA NORMAL", os.path.join(RSNA_DIR, "NORMAL"))
    evaluate_folder("RSNA PNEUMONIA", os.path.join(RSNA_DIR, "PNEUMONIA"))
    evaluate_folder("TB", TB_DIR)

    print("\n" + "=" * 72)
    print("Done")
    print("=" * 72)


if __name__ == "__main__":
    main()
