"""Calibrate TB rejection threshold for Component 2 gatekeeper."""

import glob
import os
import sys

import cv2
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from app.ml_models.component2.inference import _apply_clahe
from app.ml_models.component2.model import get_autoencoder, get_ood_shield_data

TB_DIR = r"D:\Datasets\TB\TB\TB_Chest_Radiography_Database\Tuberculosis"
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "app", "ml_models", "component2", "files")


def embed_image(image_path, encoder):
    img = cv2.imread(image_path)
    if img is None:
        return None

    img_resized = cv2.resize(img, (224, 224))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_clahe = _apply_clahe(img_rgb)
    img_ae = np.expand_dims(img_clahe, axis=0).astype("float32") / 255.0

    embedding = encoder.predict(img_ae, verbose=0)
    return embedding.flatten()


def main():
    image_paths = sorted(
        glob.glob(os.path.join(TB_DIR, "*.png"))
        + glob.glob(os.path.join(TB_DIR, "*.jpg"))
        + glob.glob(os.path.join(TB_DIR, "*.jpeg"))
    )
    if not image_paths:
        raise FileNotFoundError(f"No TB images found in {TB_DIR}")

    _, encoder = get_autoencoder()
    lung_centroid, ood_threshold = get_ood_shield_data()

    embeddings = []
    distances_to_lung = []
    failed = []

    for path in image_paths:
        embedding = embed_image(path, encoder)
        if embedding is None:
            failed.append(path)
            continue
        embeddings.append(embedding)
        distances_to_lung.append(float(np.linalg.norm(embedding - lung_centroid)))

    if not embeddings:
        raise RuntimeError("No TB embeddings could be computed.")

    embeddings = np.stack(embeddings)
    tb_centroid = embeddings.mean(axis=0)
    distances_to_tb = np.linalg.norm(embeddings - tb_centroid, axis=1)
    distances_to_lung = np.array(distances_to_lung)

    # TB images that pass the existing OOD shield (these are the ones we must block).
    ood_pass_mask = distances_to_lung <= ood_threshold
    ood_pass_distances = distances_to_lung[ood_pass_mask]

    # Reject if the image is closer to the TB centroid than to the pneumonia centroid.
    tb_centroid_distance = float(np.linalg.norm(tb_centroid - lung_centroid))
    tb_intra_max = float(distances_to_tb.max())
    tb_intra_p95 = float(np.percentile(distances_to_tb, 95))

    # Cover the full TB cluster so calibrated TB X-rays are rejected reliably.
    tb_rejection_radius = tb_intra_max

    print(f"TB images processed: {len(embeddings)}")
    print(f"Failed reads: {len(failed)}")
    print(f"OOD threshold (unchanged): {ood_threshold:.6f}")
    print(f"TB distance to lung centroid - min: {distances_to_lung.min():.4f}, "
          f"max: {distances_to_lung.max():.4f}, mean: {distances_to_lung.mean():.4f}, "
          f"p95: {np.percentile(distances_to_lung, 95):.4f}")
    print(f"TB passing current OOD shield: {ood_pass_mask.sum()} / {len(distances_to_lung)}")
    if len(ood_pass_distances):
        print(f"OOD-pass TB lung distances - min: {ood_pass_distances.min():.4f}, "
              f"max: {ood_pass_distances.max():.4f}, p95: {np.percentile(ood_pass_distances, 95):.4f}")
    print(f"TB centroid distance to lung centroid: {tb_centroid_distance:.6f}")
    print(f"TB intra-cluster p95 radius: {tb_intra_p95:.6f}")
    print(f"Chosen TB rejection radius: {tb_rejection_radius:.6f}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    np.save(os.path.join(OUTPUT_DIR, "tb_centroid.npy"), tb_centroid)
    with open(os.path.join(OUTPUT_DIR, "tb_rejection_radius.txt"), "w", encoding="utf-8") as f:
        f.write(f"{tb_rejection_radius:.8f}\n")

    print(f"Saved: {os.path.join(OUTPUT_DIR, 'tb_centroid.npy')}")
    print(f"Saved: {os.path.join(OUTPUT_DIR, 'tb_rejection_radius.txt')}")


if __name__ == "__main__":
    main()
