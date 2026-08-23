"""
Empirical bbox-convention diagnostic for component3.

There is no training script/dataset checked into this repo, so the coordinate
convention multi_task_diagnostic_model.keras actually learned for bbox_output
(xyxy? xywh? axis-swapped?) cannot be verified from source. This script runs
the real trained model against a folder of sample images and renders several
candidate interpretations of the raw bbox_output vector side by side, so a
human can visually compare them against images where the TB lesion location
is known and pick whichever candidate is consistently correct.

Usage:
    python -m app.ml_models.component3.tools.diagnose_bbox \
        --input-dir path/to/sample_images \
        [--output-dir path/to/diagnostic_output] \
        [--model-path path/to/custom_model.keras]
"""

import argparse
import glob
import os

import cv2
import numpy as np

from app.ml_models.component3.controller import DiagnosticController, CLASSES
from app.ml_models.component3.gradcam import clip_and_order_bbox

TB_CLASS_IDX = CLASSES.index("Tuberculosis")
IMAGE_EXTENSIONS = ("*.png", "*.jpg", "*.jpeg", "*.bmp")


def decode_xyxy_direct(box):
    """Current production assumption: [x_min, y_min, x_max, y_max]."""
    return clip_and_order_bbox((box[0], box[1], box[2], box[3]))


def decode_xywh_center(box):
    """[x_center, y_center, width, height] -> corners."""
    xc, yc, w, h = (float(v) for v in box)
    return clip_and_order_bbox((xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2))


def decode_xywh_topleft(box):
    """[x_min, y_min, width, height] -> corners."""
    x, y, w, h = (float(v) for v in box)
    return clip_and_order_bbox((x, y, x + w, y + h))


def decode_axis_swapped(box):
    """[y_min, x_min, y_max, x_max] (row/col transposed) -> (x_min,y_min,x_max,y_max)."""
    return clip_and_order_bbox((box[1], box[0], box[3], box[2]))


CANDIDATES = {
    "a_xyxy_direct": decode_xyxy_direct,
    "b_xywh_center": decode_xywh_center,
    "c_xywh_topleft": decode_xywh_topleft,
    "d_axis_swapped": decode_axis_swapped,
}


def _draw_candidate(display_img_rgb, box_raw, decode_fn, label):
    """Returns a copy of display_img_rgb (BGR, for cv2 drawing) with the
    decoded box and a label strip burned in."""
    h, w, _ = display_img_rgb.shape
    canvas = cv2.cvtColor(display_img_rgb, cv2.COLOR_RGB2BGR).copy()

    x_min, y_min, x_max, y_max = decode_fn(box_raw)
    px_min, py_min = int(x_min * w), int(y_min * h)
    px_max, py_max = int(x_max * w), int(y_max * h)
    cv2.rectangle(canvas, (px_min, py_min), (px_max, py_max), (0, 255, 0), 2)

    strip = np.zeros((24, w, 3), dtype=np.uint8)
    coord_text = f"{label}: ({x_min:.2f},{y_min:.2f},{x_max:.2f},{y_max:.2f})"
    cv2.putText(strip, coord_text, (4, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

    return np.vstack([strip, canvas])


def render_candidates(display_img_rgb, box_raw, output_dir, stem):
    """Saves one PNG per candidate plus a combined 2x2 grid, all in output_dir."""
    tiles = []
    for name, decode_fn in CANDIDATES.items():
        tile = _draw_candidate(display_img_rgb, box_raw, decode_fn, name)
        cv2.imwrite(os.path.join(output_dir, f"{stem}__{name}.png"), tile)
        tiles.append(tile)

    top_row = np.hstack(tiles[0:2])
    bottom_row = np.hstack(tiles[2:4])
    grid = np.vstack([top_row, bottom_row])
    cv2.imwrite(os.path.join(output_dir, f"{stem}__all_candidates.png"), grid)


def _print_stats_table(title, bboxes):
    print(f"\n{title} (n={len(bboxes)})")
    if not bboxes:
        print("  (no samples)")
        return
    arr = np.array(bboxes, dtype=np.float32)
    means = arr.mean(axis=0)
    stds = arr.std(axis=0)
    labels = ["coord0", "coord1", "coord2", "coord3"]
    for label, mean, std in zip(labels, means, stds):
        flag = "  <-- LOW VARIANCE (possible regression-to-mean)" if std < 0.03 else ""
        print(f"  {label}: mean={mean:.4f}  std={std:.4f}{flag}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, help="Folder of sample chest X-ray images.")
    parser.add_argument("--output-dir", default=None, help="Where to write candidate renders.")
    parser.add_argument("--model-path", default=None, help="Optional override for the .keras model path.")
    args = parser.parse_args()

    input_dir = os.path.abspath(args.input_dir)
    output_dir = args.output_dir or os.path.join(os.path.dirname(input_dir), "diagnose_bbox_output")
    os.makedirs(output_dir, exist_ok=True)

    controller = DiagnosticController(model_path=args.model_path)

    image_paths = []
    for pattern in IMAGE_EXTENSIONS:
        image_paths.extend(glob.glob(os.path.join(input_dir, pattern)))
    image_paths.sort()

    if not image_paths:
        print(f"No images found in {input_dir} (looked for {IMAGE_EXTENSIONS}).")
        return

    all_bboxes = []
    tb_bboxes = []
    tb_count = 0

    for path in image_paths:
        with open(path, "rb") as f:
            image_bytes = f.read()

        display_img, img_tensor = controller.explain_engine.preprocess_for_explainability(image_bytes)
        predictions = controller.diagnostic_model.predict(img_tensor, verbose=0)
        class_probs = predictions["class_output"][0]
        bbox_raw = predictions["bbox_output"][0]
        all_bboxes.append(bbox_raw)

        pred_idx = int(np.argmax(class_probs))
        stem = os.path.splitext(os.path.basename(path))[0]

        if pred_idx == TB_CLASS_IDX:
            tb_count += 1
            tb_bboxes.append(bbox_raw)
            render_candidates(display_img, bbox_raw, output_dir, stem)
            print(f"TB   {stem}: raw_bbox={np.round(bbox_raw, 4).tolist()} -> renders written")
        else:
            print(f"SKIP {stem}: predicted {CLASSES[pred_idx]} (not TB), no box rendered")

    _print_stats_table("Aggregate stats: TB-predicted images only", tb_bboxes)
    _print_stats_table("Aggregate stats: all images", all_bboxes)

    print(f"\nProcessed {len(image_paths)} images, {tb_count} predicted TB.")
    print(f"Candidate renders written to: {output_dir}")


if __name__ == "__main__":
    main()
