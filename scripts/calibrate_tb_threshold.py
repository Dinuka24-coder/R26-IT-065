"""Calibrate TB shield using TB dataset and pneumonia safety guard sets."""

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
PEDIATRIC_TEST_DIR = r"D:\Datasets\Pneumonia\archive\chest_xray\test"
RSNA_DIR = r"D:\Datasets\Pneumonia\rsna_1000_samples"
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "app", "ml_models", "component2", "files")


def list_images(folder):
    paths = set()
    for ext in ("*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG", "*.JPEG"):
        paths.update(glob.glob(os.path.join(folder, ext)))
    return sorted(paths)


def embed_paths(paths, encoder):
    embeddings = []
    kept = []
    for path in paths:
        img = cv2.imread(path)
        if img is None:
            continue
        img_resized = cv2.resize(img, (224, 224))
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_clahe = _apply_clahe(img_rgb)
        img_ae = np.expand_dims(img_clahe, axis=0).astype("float32") / 255.0
        emb = encoder.predict(img_ae, verbose=0).flatten()
        embeddings.append(emb)
        kept.append(path)
    return np.array(embeddings), kept


def tb_shield_would_reject(tb_dist, lung_dist, radius):
    return tb_dist <= radius and lung_dist > tb_dist


def main():
    print("Loading encoder...")
    _, encoder = get_autoencoder()
    lung_centroid, ood_threshold = get_ood_shield_data()

    tb_paths = list_images(TB_DIR)
    guard_paths = []
    for label in ("NORMAL", "PNEUMONIA"):
        guard_paths.extend(list_images(os.path.join(PEDIATRIC_TEST_DIR, label)))
        guard_paths.extend(list_images(os.path.join(RSNA_DIR, label)))

    print(f"TB images: {len(tb_paths)}")
    print(f"Pneumonia guard images (pediatric + RSNA, normal + pneumonia): {len(guard_paths)}")

    print("\nEmbedding TB images...")
    tb_embeddings, _ = embed_paths(tb_paths, encoder)
    print("Embedding pneumonia guard images...")
    guard_embeddings, guard_kept = embed_paths(guard_paths, encoder)

    tb_centroid = tb_embeddings.mean(axis=0)
    tb_distances = np.linalg.norm(tb_embeddings - tb_centroid, axis=1)
    tb_intra_max = float(tb_distances.max())

    guard_tb_dist = np.linalg.norm(guard_embeddings - tb_centroid, axis=1)
    guard_lung_dist = np.linalg.norm(guard_embeddings - lung_centroid, axis=1)

    ood_pass_mask = guard_lung_dist <= ood_threshold
    ood_pass_tb = guard_tb_dist[ood_pass_mask]
    ood_pass_lung = guard_lung_dist[ood_pass_mask]
    print(f"Pneumonia guard images passing OOD: {ood_pass_mask.sum()}/{len(guard_embeddings)}")

    # Largest radius that keeps every OOD-passing pneumonia image passing the TB shield.
    vulnerable = ood_pass_tb[ood_pass_lung > ood_pass_tb]
    if len(vulnerable) == 0:
        safe_radius = tb_intra_max
        print("\nNo guard images are closer to TB than pneumonia centroid; using full TB cluster radius.")
    else:
        safe_radius = float(vulnerable.min()) * 0.999
        print(f"\nGuard images (OOD-pass) closer to TB than pneumonia: {len(vulnerable)}/{ood_pass_mask.sum()}")

    false_rejects = sum(
        1
        for tb_d, lung_d in zip(ood_pass_tb, ood_pass_lung)
        if tb_shield_would_reject(float(tb_d), float(lung_d), safe_radius)
    )
    tb_rejects = sum(
        1
        for emb in tb_embeddings
        for tb_d, lung_d in [
            (
                float(np.linalg.norm(emb - tb_centroid)),
                float(np.linalg.norm(emb - lung_centroid)),
            )
        ]
        if tb_shield_would_reject(tb_d, lung_d, safe_radius)
    )
    tb_ood_rejects = sum(
        1
        for emb in tb_embeddings
        if float(np.linalg.norm(emb - lung_centroid)) > ood_threshold
    )

    print("\n=== Calibration results ===")
    print(f"TB cluster max intra-distance: {tb_intra_max:.6f}")
    print(f"Pneumonia-safe radius (0 false rejects): {safe_radius:.6f}")
    print(f"Pneumonia guard false rejects at safe radius: {false_rejects}/{ood_pass_mask.sum()}")

    # Use full TB cluster radius so TB images are rejected reliably.
    chosen_radius = tb_intra_max
    tb_rejects_full = sum(
        1
        for emb in tb_embeddings
        for tb_d, lung_d in [
            (
                float(np.linalg.norm(emb - tb_centroid)),
                float(np.linalg.norm(emb - lung_centroid)),
            )
        ]
        if tb_shield_would_reject(tb_d, lung_d, chosen_radius)
    )
    pneumonia_false_full = sum(
        1
        for tb_d, lung_d in zip(ood_pass_tb, ood_pass_lung)
        if tb_shield_would_reject(float(tb_d), float(lung_d), chosen_radius)
    )

    print(f"Chosen TB rejection radius (full cluster): {chosen_radius:.6f}")
    print(f"TB rejected by TB shield at chosen radius: {tb_rejects_full}/{len(tb_embeddings)}")
    print(f"Pneumonia guard false rejects at chosen radius: {pneumonia_false_full}/{ood_pass_mask.sum()}")
    print(f"TB rejected by OOD only: {tb_ood_rejects}/{len(tb_embeddings)}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    np.save(os.path.join(OUTPUT_DIR, "tb_centroid.npy"), tb_centroid)
    with open(os.path.join(OUTPUT_DIR, "tb_rejection_radius.txt"), "w", encoding="utf-8") as f:
        f.write(f"{chosen_radius:.8f}\n")

    print(f"\nSaved: {os.path.join(OUTPUT_DIR, 'tb_centroid.npy')}")
    print(f"Saved: {os.path.join(OUTPUT_DIR, 'tb_rejection_radius.txt')}")


if __name__ == "__main__":
    main()
