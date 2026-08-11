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


def main():
    print("=" * 60)
    print("Component 2 Shield Tests")
    print("=" * 60)

    # 1. Random noise should fail (OOD)
    print("\n[1] Random noise image")
    noise = np.random.randint(0, 256, (300, 300, 3), dtype=np.uint8)
    try:
        run_pneumonia_inference(noise)
        print("  FAIL - expected InvalidXRayError")
    except InvalidXRayError:
        print("  PASS - rejected as invalid X-ray")

    # 2. Pneumonia centroid embedding should pass both shields
    print("\n[2] Simulated pneumonia embedding (lung centroid)")
    lung_centroid, _ = get_ood_shield_data()
    outcome, d_lung, d_tb = tb_shield_would_reject(lung_centroid)
    if outcome == "pass":
        print(f"  PASS - shields allow pneumonia centroid (lung={d_lung:.4f}, tb={d_tb:.4f})")
    else:
        print(f"  FAIL - unexpectedly {outcome} (lung={d_lung:.4f}, tb={d_tb:.4f})")

    # 3. TB centroid embedding should be rejected by TB shield
    print("\n[3] Simulated TB embedding (TB centroid)")
    tb_centroid, _ = get_tb_shield_data()
    outcome, d_lung, d_tb = tb_shield_would_reject(tb_centroid)
    if outcome == "tb_reject":
        print(f"  PASS - TB shield rejects TB centroid (lung={d_lung:.4f}, tb={d_tb:.4f})")
    else:
        print(f"  FAIL - expected tb_reject, got {outcome}")

    # 4. Real TB images via full inference pipeline
    print("\n[4] Real TB X-rays (full inference pipeline)")
    tb_paths = sorted(glob.glob(os.path.join(TB_DIR, "*.png")))[:20]
    if not tb_paths:
        print("  SKIP - TB dataset not found")
    else:
        rejected = 0
        passed = 0
        for path in tb_paths:
            img = cv2.imread(path)
            name = os.path.basename(path)
            try:
                run_pneumonia_inference(img)
                passed += 1
                print(f"  FAIL - {name} was NOT rejected")
            except InvalidXRayError:
                rejected += 1
        print(f"  Sample result: {rejected}/{len(tb_paths)} TB images rejected")
        if rejected == len(tb_paths):
            print("  PASS - all sampled TB images rejected")
        elif rejected >= len(tb_paths) * 0.9:
            print("  PASS - most sampled TB images rejected")
        else:
            print("  WARN - some TB images passed through")

    # 5. Full TB dataset shield stats (fast path, no classifier)
    print("\n[5] Full TB dataset shield analysis (700 images)")
    all_tb = sorted(glob.glob(os.path.join(TB_DIR, "*.png")))
    if not all_tb:
        print("  SKIP - TB dataset not found")
    else:
        ood_reject = tb_reject = pass_through = 0
        for path in all_tb:
            emb = embed_image(path)
            if emb is None:
                continue
            outcome, _, _ = tb_shield_would_reject(emb)
            if outcome == "ood_reject":
                ood_reject += 1
            elif outcome == "tb_reject":
                tb_reject += 1
            else:
                pass_through += 1

        total = len(all_tb)
        print(f"  OOD rejects:      {ood_reject}/{total}")
        print(f"  TB shield rejects:{tb_reject}/{total}")
        print(f"  Would pass both:  {pass_through}/{total}")
        if pass_through <= 11:
            print("  PASS - TB shield blocks nearly all TB images")
        else:
            print("  WARN - more TB images pass than expected")

    print("\n" + "=" * 60)
    print("Done")
    print("=" * 60)


if __name__ == "__main__":
    main()
