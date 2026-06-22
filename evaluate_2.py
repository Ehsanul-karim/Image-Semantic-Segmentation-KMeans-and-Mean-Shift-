#!/usr/bin/env python3
"""
Segmentation evaluation with label matching, optional LabelMe JSON support,
CSV reporting, and visual outputs.

Expected tree
-------------
ROOT/
  GT Ehsanul Karim/
    Img01/
      mask.png
      contour.png
      contour_overlay.png
      Img01.json
    Img02/
      mask.png
      Img02.json
  Img01/
    hsv/
      KMeans/
        Post Processed*.png
      Half Resize PCA 95 Mean Shift/
        Postprocessed*.png
    ...
  Img02/
    ...

What is evaluated
-----------------
Only generated mask images whose filename starts with:
  - Post Processed
  - Postprocessed

The script ignores non-mask GT helper files such as:
  - contour.png
  - contour_overlay.png

GT source options
-----------------
--gt-source mask
    Use GT Ehsanul Karim/ImgXX/mask.png as the actual ground truth.

--gt-source json
    Rasterize GT Ehsanul Karim/ImgXX/ImgXX.json into a semantic mask
    and use that as the actual ground truth.

--gt-source auto
    Use mask.png when available; otherwise use ImgXX.json.

--name-gt-classes-from-json
    When using mask.png, also read ImgXX.json and infer semantic class names
    such as Starfish / Stone by overlap with the mask.png labels.

Outputs
-------
output/
  per_image_results.csv
  per_class_results.csv
  confusion_matrices_long.csv
  label_matches.csv
  qualitative_results.csv
  visuals/*.png

performance/
  macro_by_method.csv
  macro_by_algorithm.csv
  macro_by_feature_algorithm.csv
  overall_macro_average.csv
  per_image_results.csv
  per_class_results.csv
  metric_bar_*.png
  method_metric_heatmap.png
  confusion_matrices/*.png
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

try:
    from scipy.optimize import linear_sum_assignment
    from scipy import ndimage as ndi
except Exception as exc:  # pragma: no cover
    raise SystemExit(
        "This script needs scipy. Install requirements with:\n"
        "  pip install numpy pandas pillow scipy matplotlib\n"
    ) from exc


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
POST_PROCESSED_RE = re.compile(r"^post\s*processed", re.IGNORECASE)
IMG_DIR_RE = re.compile(r"^Img\d+$", re.IGNORECASE)


try:
    NEAREST = Image.Resampling.NEAREST
except AttributeError:  # Pillow < 9
    NEAREST = Image.NEAREST


@dataclass
class PredictionItem:
    image_id: str
    gt_mask_path: Optional[Path]
    gt_json_path: Optional[Path]
    pred_path: Path
    rel_path: str
    feature_set: str
    method_dir: str
    algorithm: str
    prediction_name: str

    @property
    def method_key(self) -> str:
        return f"{self.feature_set} | {self.method_dir} | {self.prediction_name}"


@dataclass
class GTData:
    labels: np.ndarray
    class_names: List[str]
    raw_rgb: np.ndarray
    source_type: str
    source_path: Path


def natural_key(text: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", str(text))]


def safe_name(text: str, max_len: int = 180) -> str:
    text = re.sub(r"[^\w\-.]+", "__", text.strip(), flags=re.UNICODE)
    text = text.strip("._")
    return text[:max_len] if len(text) > max_len else text


def load_pil(path: Path) -> Image.Image:
    try:
        return Image.open(path)
    except Exception as exc:
        raise RuntimeError(f"Could not read image: {path}") from exc


def load_mask_as_labels(path: Path) -> Tuple[np.ndarray, List[str], np.ndarray]:
    """
    Load a segmentation mask and convert unique grayscale values or RGB colors
    to compact labels 0..K-1.

    Returns:
      labels: HxW int array
      label_names: readable original values, e.g. "0" or "#ff0000"
      raw_rgb: HxWx3 uint8 image for visualization
    """
    img = load_pil(path)
    arr = np.array(img)

    if arr.ndim == 2:
        flat_values = arr.reshape(-1)
        unique, inverse = np.unique(flat_values, return_inverse=True)
        label_names = [str(int(v)) for v in unique]
        labels = inverse.reshape(arr.shape).astype(np.int32)
        raw_rgb = np.array(img.convert("RGB"))
        return labels, label_names, raw_rgb

    if arr.ndim == 3:
        # Ignore alpha if present.
        rgb = arr[:, :, :3].astype(np.uint32)
        packed = (rgb[:, :, 0] << 16) + (rgb[:, :, 1] << 8) + rgb[:, :, 2]
        unique, inverse = np.unique(packed.reshape(-1), return_inverse=True)
        label_names = [
            f"#{int((v >> 16) & 255):02x}{int((v >> 8) & 255):02x}{int(v & 255):02x}"
            for v in unique
        ]
        labels = inverse.reshape(packed.shape).astype(np.int32)
        raw_rgb = np.array(img.convert("RGB"))
        return labels, label_names, raw_rgb

    raise ValueError(f"Unsupported mask shape {arr.shape} for {path}")


def rasterize_labelme_json(json_path: Path) -> Tuple[np.ndarray, List[str], np.ndarray]:
    """
    Rasterize a LabelMe JSON file into integer semantic labels.

    Label 0 is background.
    Same text label appearing in multiple shapes receives the same class id.
    Later shapes overwrite earlier ones, matching common LabelMe mask behavior.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    height = int(data["imageHeight"])
    width = int(data["imageWidth"])

    label_to_id: Dict[str, int] = {"background": 0}
    mask_img = Image.new("I", (width, height), 0)
    draw = ImageDraw.Draw(mask_img)

    for shape in data.get("shapes", []):
        label = str(shape.get("label", "")).strip() or "unlabeled"
        if label not in label_to_id:
            label_to_id[label] = len(label_to_id)

        class_id = label_to_id[label]
        points = shape.get("points", [])
        shape_type = str(shape.get("shape_type", "polygon")).lower()

        if not points:
            continue

        xy = [(float(x), float(y)) for x, y in points]

        if shape_type == "polygon":
            if len(xy) >= 3:
                draw.polygon(xy, fill=int(class_id))
        elif shape_type == "rectangle":
            if len(xy) >= 2:
                x0, y0 = xy[0]
                x1, y1 = xy[1]
                draw.rectangle([x0, y0, x1, y1], fill=int(class_id))
        elif shape_type == "circle":
            if len(xy) >= 2:
                x0, y0 = xy[0]
                x1, y1 = xy[1]
                radius = ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5
                draw.ellipse([x0 - radius, y0 - radius, x0 + radius, y0 + radius], fill=int(class_id))
        elif shape_type == "line":
            # Lines are not regions. Draw as width 1 only so they are represented if present.
            if len(xy) >= 2:
                draw.line(xy, fill=int(class_id), width=1)
        elif shape_type == "point":
            x0, y0 = xy[0]
            draw.ellipse([x0 - 1, y0 - 1, x0 + 1, y0 + 1], fill=int(class_id))
        else:
            # Fallback: treat unknown shapes with >=3 points as polygons.
            if len(xy) >= 3:
                draw.polygon(xy, fill=int(class_id))

    labels = np.array(mask_img, dtype=np.int32)
    class_names = [None] * len(label_to_id)
    for name, idx in label_to_id.items():
        class_names[idx] = name
    class_names = [str(x) for x in class_names]

    raw_rgb = colorize(labels)
    return labels, class_names, raw_rgb


def resize_labels_nearest(labels: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
    if labels.shape == target_shape:
        return labels

    pil = Image.fromarray(labels.astype(np.int32), mode="I")
    resized = pil.resize((target_shape[1], target_shape[0]), resample=NEAREST)
    return np.array(resized).astype(np.int32)


def resize_rgb_nearest(rgb: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
    if rgb.shape[:2] == target_shape:
        return rgb

    pil = Image.fromarray(rgb.astype(np.uint8), mode="RGB")
    resized = pil.resize((target_shape[1], target_shape[0]), resample=NEAREST)
    return np.array(resized).astype(np.uint8)


def compact_labels(labels: np.ndarray) -> Tuple[np.ndarray, List[int]]:
    unique, inverse = np.unique(labels.reshape(-1), return_inverse=True)
    return inverse.reshape(labels.shape).astype(np.int32), [int(v) for v in unique]


def contingency_matrix(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    g = int(gt.max()) + 1 if gt.size else 0
    p = int(pred.max()) + 1 if pred.size else 0
    idx = gt.reshape(-1) * p + pred.reshape(-1)
    return np.bincount(idx, minlength=g * p).reshape(g, p).astype(np.int64)


def infer_gt_names_from_json_mask(
    gt_labels: np.ndarray,
    gt_value_names: Sequence[str],
    json_path: Optional[Path],
    min_overlap_pixels: int = 10,
) -> List[str]:
    """
    If GT is mask.png and LabelMe JSON exists, infer semantic class names for
    the GT mask labels by overlap.

    This is useful when mask.png contains colors/values but JSON has labels like
    Starfish and Stone.
    """
    if json_path is None or not json_path.exists():
        return list(gt_value_names)

    try:
        json_labels, json_names, _ = rasterize_labelme_json(json_path)
        json_labels = resize_labels_nearest(json_labels, gt_labels.shape)
    except Exception as exc:
        print(f"[WARN] Could not use JSON class names from {json_path}: {exc}", file=sys.stderr)
        return list(gt_value_names)

    cm = contingency_matrix(gt_labels, json_labels)
    inferred: List[str] = []

    for gt_id in range(cm.shape[0]):
        overlaps = cm[gt_id].copy()
        best_json_id = int(np.argmax(overlaps))
        best_overlap = int(overlaps[best_json_id])

        # Keep background if the best overlapping JSON label is background.
        # Otherwise append the original mask value for traceability.
        if best_overlap >= min_overlap_pixels:
            semantic = json_names[best_json_id]
            if semantic == "background":
                inferred.append(f"background_or_{gt_value_names[gt_id]}")
            else:
                inferred.append(f"{semantic} ({gt_value_names[gt_id]})")
        else:
            inferred.append(str(gt_value_names[gt_id]))

    return inferred


def load_gt_data(
    image_id: str,
    gt_mask_path: Optional[Path],
    gt_json_path: Optional[Path],
    gt_source: str,
    name_gt_classes_from_json: bool,
) -> GTData:
    """
    Load GT from mask.png or LabelMe JSON based on gt_source.
    """
    selected_source = gt_source

    if gt_source == "auto":
        if gt_mask_path is not None and gt_mask_path.exists():
            selected_source = "mask"
        elif gt_json_path is not None and gt_json_path.exists():
            selected_source = "json"
        else:
            raise FileNotFoundError(f"No GT mask.png or JSON found for {image_id}")

    if selected_source == "mask":
        if gt_mask_path is None or not gt_mask_path.exists():
            raise FileNotFoundError(f"GT mask.png not found for {image_id}: {gt_mask_path}")
        labels, gt_value_names, raw_rgb = load_mask_as_labels(gt_mask_path)
        if name_gt_classes_from_json:
            class_names = infer_gt_names_from_json_mask(labels, gt_value_names, gt_json_path)
        else:
            class_names = list(gt_value_names)
        return GTData(labels=labels, class_names=class_names, raw_rgb=raw_rgb, source_type="mask", source_path=gt_mask_path)

    if selected_source == "json":
        if gt_json_path is None or not gt_json_path.exists():
            raise FileNotFoundError(f"GT JSON not found for {image_id}: {gt_json_path}")
        labels, class_names, raw_rgb = rasterize_labelme_json(gt_json_path)
        return GTData(labels=labels, class_names=class_names, raw_rgb=raw_rgb, source_type="json", source_path=gt_json_path)

    raise ValueError(f"Unknown GT source: {gt_source}")


def match_prediction_labels(
    gt: np.ndarray,
    pred: np.ndarray,
    gt_names: Sequence[str],
    pred_names: Sequence[str],
    matching: str = "hungarian",
) -> Tuple[np.ndarray, np.ndarray, List[str], List[Dict[str, object]]]:
    """
    Remap prediction labels to GT label ids.

    Hungarian mode uses one-to-one matching that maximizes total pixel overlap.
    many_to_one mode maps each predicted label to the GT label with maximum overlap.
    """
    cm = contingency_matrix(gt, pred)
    n_gt, n_pred = cm.shape

    pred_to_new = np.full(n_pred, -1, dtype=np.int32)
    match_rows: List[Dict[str, object]] = []

    if matching == "many_to_one":
        for pred_id in range(n_pred):
            gt_id = int(np.argmax(cm[:, pred_id]))
            pred_to_new[pred_id] = gt_id
            match_rows.append(
                {
                    "pred_label_id": pred_id,
                    "pred_label_value": pred_names[pred_id] if pred_id < len(pred_names) else str(pred_id),
                    "matched_gt_label_id": gt_id,
                    "matched_gt_label_value": gt_names[gt_id] if gt_id < len(gt_names) else str(gt_id),
                    "overlap_pixels": int(cm[gt_id, pred_id]),
                    "is_extra_prediction_label": False,
                }
            )
    else:
        row_ind, col_ind = linear_sum_assignment(-cm)
        for gt_id, pred_id in zip(row_ind, col_ind):
            gt_id = int(gt_id)
            pred_id = int(pred_id)
            if gt_id < n_gt and pred_id < n_pred:
                pred_to_new[pred_id] = gt_id
                match_rows.append(
                    {
                        "pred_label_id": pred_id,
                        "pred_label_value": pred_names[pred_id] if pred_id < len(pred_names) else str(pred_id),
                        "matched_gt_label_id": gt_id,
                        "matched_gt_label_value": gt_names[gt_id] if gt_id < len(gt_names) else str(gt_id),
                        "overlap_pixels": int(cm[gt_id, pred_id]),
                        "is_extra_prediction_label": False,
                    }
                )

        next_extra = n_gt
        for pred_id in range(n_pred):
            if pred_to_new[pred_id] < 0:
                pred_to_new[pred_id] = next_extra
                best_gt = int(np.argmax(cm[:, pred_id]))
                match_rows.append(
                    {
                        "pred_label_id": pred_id,
                        "pred_label_value": pred_names[pred_id] if pred_id < len(pred_names) else str(pred_id),
                        "matched_gt_label_id": next_extra,
                        "matched_gt_label_value": f"extra_pred_{pred_id}",
                        "overlap_pixels": int(cm[best_gt, pred_id]),
                        "is_extra_prediction_label": True,
                    }
                )
                next_extra += 1

    remapped = pred_to_new[pred]
    n_cols = int(max(n_gt, int(remapped.max()) + 1))
    idx = gt.reshape(-1) * n_cols + remapped.reshape(-1)
    matched_cm = np.bincount(idx, minlength=n_gt * n_cols).reshape(n_gt, n_cols).astype(np.int64)

    column_names = list(gt_names)
    for col_id in range(n_gt, n_cols):
        column_names.append(f"extra_pred_{col_id - n_gt}")

    return remapped.astype(np.int32), matched_cm, column_names, match_rows


def safe_div(num: float, den: float, zero_value: float = 0.0) -> float:
    return float(num / den) if den else float(zero_value)


def boundary_map(mask: np.ndarray) -> np.ndarray:
    if not np.any(mask):
        return np.zeros(mask.shape, dtype=bool)
    structure = np.ones((3, 3), dtype=bool)
    eroded = ndi.binary_erosion(mask.astype(bool), structure=structure, border_value=0)
    return mask.astype(bool) & ~eroded


def boundary_f1(gt_binary: np.ndarray, pred_binary: np.ndarray, tolerance: int) -> float:
    gt_b = boundary_map(gt_binary)
    pred_b = boundary_map(pred_binary)

    n_gt = int(gt_b.sum())
    n_pred = int(pred_b.sum())

    if n_gt == 0 and n_pred == 0:
        return 1.0
    if n_gt == 0 or n_pred == 0:
        return 0.0

    gt_dil = ndi.binary_dilation(gt_b, iterations=tolerance)
    pred_dil = ndi.binary_dilation(pred_b, iterations=tolerance)

    precision = safe_div(int((pred_b & gt_dil).sum()), n_pred)
    recall = safe_div(int((gt_b & pred_dil).sum()), n_gt)
    return safe_div(2.0 * precision * recall, precision + recall)


def compute_metrics(
    gt: np.ndarray,
    remapped_pred: np.ndarray,
    matched_cm: np.ndarray,
    gt_names: Sequence[str],
    tolerance: int,
    include_background: bool,
) -> Tuple[Dict[str, float], List[Dict[str, object]]]:
    n_gt = len(gt_names)
    total = int(matched_cm.sum())

    per_class_rows: List[Dict[str, object]] = []
    precisions = []
    recalls = []
    ious = []
    bfs = []

    weighted_p = 0.0
    weighted_r = 0.0
    weighted_iou = 0.0
    weighted_bf = 0.0
    weighted_total_support = 0

    for class_id in range(n_gt):
        tp = int(matched_cm[class_id, class_id]) if class_id < matched_cm.shape[1] else 0
        fp = int(matched_cm[:, class_id].sum() - tp) if class_id < matched_cm.shape[1] else 0
        fn = int(matched_cm[class_id, :].sum() - tp)
        support = int(matched_cm[class_id, :].sum())

        precision = safe_div(tp, tp + fp)
        recall = safe_div(tp, tp + fn)
        iou = safe_div(tp, tp + fp + fn)
        bf = boundary_f1(gt == class_id, remapped_pred == class_id, tolerance=tolerance)

        use_for_macro = include_background or class_id != 0
        if use_for_macro:
            precisions.append(precision)
            recalls.append(recall)
            ious.append(iou)
            bfs.append(bf)

            weighted_p += support * precision
            weighted_r += support * recall
            weighted_iou += support * iou
            weighted_bf += support * bf
            weighted_total_support += support

        per_class_rows.append(
            {
                "class_id": class_id,
                "class_name": gt_names[class_id],
                "support_pixels": support,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": precision,
                "recall": recall,
                "iou": iou,
                "boundary_f1": bf,
                "included_in_macro_average": use_for_macro,
            }
        )

    diagonal_cols = min(n_gt, matched_cm.shape[1])
    accuracy_count = int(sum(matched_cm[i, i] for i in range(diagonal_cols)))

    summary = {
        "pixel_accuracy": safe_div(accuracy_count, total),
        "macro_precision": float(np.mean(precisions)) if precisions else np.nan,
        "macro_recall": float(np.mean(recalls)) if recalls else np.nan,
        "macro_iou": float(np.mean(ious)) if ious else np.nan,
        "macro_boundary_f1": float(np.mean(bfs)) if bfs else np.nan,
        "weighted_precision": safe_div(weighted_p, weighted_total_support),
        "weighted_recall": safe_div(weighted_r, weighted_total_support),
        "weighted_iou": safe_div(weighted_iou, weighted_total_support),
        "weighted_boundary_f1": safe_div(weighted_bf, weighted_total_support),
    }
    return summary, per_class_rows


def colorize(labels: np.ndarray) -> np.ndarray:
    labels = labels.astype(np.int32)
    n = int(labels.max()) + 1 if labels.size else 1

    # Deterministic palette; label 0 is black.
    rng = np.random.default_rng(12345)
    palette = rng.integers(30, 245, size=(max(n, 1), 3), dtype=np.uint8)
    palette[0] = np.array([0, 0, 0], dtype=np.uint8)
    return palette[np.clip(labels, 0, n - 1)]


def save_visual_panel(
    out_path: Path,
    image_id: str,
    method_key: str,
    gt_raw: np.ndarray,
    pred_raw: np.ndarray,
    gt_labels: np.ndarray,
    remapped_pred: np.ndarray,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    error = gt_labels != remapped_pred
    error_rgb = np.zeros((*error.shape, 3), dtype=np.uint8)
    error_rgb[error] = np.array([255, 0, 0], dtype=np.uint8)

    panels = [
        ("GT raw", gt_raw),
        ("Prediction raw", pred_raw),
        ("GT labels", colorize(gt_labels)),
        ("Matched prediction", colorize(remapped_pred)),
        ("Error pixels", error_rgb),
    ]

    fig, axes = plt.subplots(1, len(panels), figsize=(4 * len(panels), 4))
    if len(panels) == 1:
        axes = [axes]

    for ax, (title, im) in zip(axes, panels):
        ax.imshow(im)
        ax.set_title(title, fontsize=9)
        ax.axis("off")

    fig.suptitle(f"{image_id} | {method_key}", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_confusion_heatmap(
    out_path: Path,
    cm_norm: np.ndarray,
    row_names: Sequence[str],
    col_names: Sequence[str],
    title: str,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    max_labels_to_show = 40
    cm_plot = cm_norm
    rnames = list(row_names)
    cnames = list(col_names)

    if cm_norm.shape[0] > max_labels_to_show or cm_norm.shape[1] > max_labels_to_show:
        cm_plot = cm_norm[:max_labels_to_show, :max_labels_to_show]
        rnames = rnames[:max_labels_to_show]
        cnames = cnames[:max_labels_to_show]
        title += f" (first {max_labels_to_show} labels shown)"

    fig_w = max(6, 0.35 * len(cnames))
    fig_h = max(5, 0.35 * len(rnames))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(cm_plot, vmin=0, vmax=1, interpolation="nearest")
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("Predicted label after matching")
    ax.set_ylabel("GT label")
    ax.set_xticks(np.arange(len(cnames)))
    ax.set_yticks(np.arange(len(rnames)))
    ax.set_xticklabels(cnames, rotation=90, fontsize=7)
    ax.set_yticklabels(rnames, fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def discover_predictions(root: Path, gt_folder: str) -> List[PredictionItem]:
    gt_root = root / gt_folder
    if not gt_root.exists():
        raise FileNotFoundError(f"GT folder not found: {gt_root}")

    items: List[PredictionItem] = []
    image_dirs = sorted(
        [p for p in root.iterdir() if p.is_dir() and IMG_DIR_RE.match(p.name)],
        key=lambda p: natural_key(p.name),
    )

    for img_dir in image_dirs:
        image_id = img_dir.name
        gt_image_dir = gt_root / image_id
        gt_mask_path = gt_image_dir / "mask.png"
        gt_json_path = gt_image_dir / f"{image_id}.json"

        if not gt_mask_path.exists() and not gt_json_path.exists():
            print(f"[WARN] Missing both GT mask and JSON, skipping {image_id}: {gt_image_dir}", file=sys.stderr)
            continue

        for pred_path in sorted(img_dir.rglob("*"), key=lambda p: natural_key(str(p))):
            if not pred_path.is_file():
                continue
            if pred_path.suffix.lower() not in IMAGE_EXTS:
                continue
            if not POST_PROCESSED_RE.match(pred_path.name):
                continue

            rel_parts = pred_path.relative_to(img_dir).parts
            rel_path = str(pred_path.relative_to(root))

            method_idx = None
            algorithm = ""
            for idx, part in enumerate(rel_parts[:-1]):
                low = part.lower()
                if "kmeans" in low:
                    method_idx = idx
                    algorithm = "KMeans"
                    break
                if "mean shift" in low or "meanshift" in low:
                    method_idx = idx
                    algorithm = "Mean Shift"
                    break

            if method_idx is None:
                print(f"[WARN] Postprocessed file is not under KMeans/Mean Shift folder, skipping: {pred_path}", file=sys.stderr)
                continue

            feature_set = "__".join(rel_parts[:method_idx]) if method_idx > 0 else "unknown_feature"
            method_dir = rel_parts[method_idx]

            items.append(
                PredictionItem(
                    image_id=image_id,
                    gt_mask_path=gt_mask_path if gt_mask_path.exists() else None,
                    gt_json_path=gt_json_path if gt_json_path.exists() else None,
                    pred_path=pred_path,
                    rel_path=rel_path,
                    feature_set=feature_set,
                    method_dir=method_dir,
                    algorithm=algorithm,
                    prediction_name=pred_path.stem,
                )
            )

    return items


def row_normalized(cm: np.ndarray) -> np.ndarray:
    denom = cm.sum(axis=1, keepdims=True)
    return np.divide(cm, denom, out=np.zeros_like(cm, dtype=float), where=denom != 0)


def plot_group_summary(
    df: pd.DataFrame,
    group_cols: List[str],
    out_csv: Path,
) -> pd.DataFrame:
    metrics = [
        "macro_precision",
        "macro_recall",
        "macro_iou",
        "macro_boundary_f1",
        "pixel_accuracy",
    ]

    summary = (
        df.groupby(group_cols, dropna=False)
        .agg(
            n_images=("image_id", "nunique"),
            n_predictions=("prediction_path", "count"),
            **{m: (m, "mean") for m in metrics},
        )
        .reset_index()
        .sort_values(["macro_iou", "macro_boundary_f1"], ascending=False)
    )
    summary.to_csv(out_csv, index=False)
    return summary


def plot_performance_summaries(per_image_df: pd.DataFrame, perf_dir: Path, top_n: int) -> pd.DataFrame:
    perf_dir.mkdir(parents=True, exist_ok=True)

    macro_by_method = plot_group_summary(
        per_image_df,
        ["method_key", "feature_set", "method_dir", "algorithm"],
        perf_dir / "macro_by_method.csv",
    )

    plot_group_summary(
        per_image_df,
        ["algorithm"],
        perf_dir / "macro_by_algorithm.csv",
    )

    plot_group_summary(
        per_image_df,
        ["feature_set", "algorithm"],
        perf_dir / "macro_by_feature_algorithm.csv",
    )

    metrics = [
        "macro_precision",
        "macro_recall",
        "macro_iou",
        "macro_boundary_f1",
        "pixel_accuracy",
    ]

    overall = pd.DataFrame(
        [
            {
                "n_predictions": len(per_image_df),
                "n_images": per_image_df["image_id"].nunique(),
                **{m: per_image_df[m].mean() for m in metrics},
            }
        ]
    )
    overall.to_csv(perf_dir / "overall_macro_average.csv", index=False)

    plot_df = macro_by_method.head(top_n).copy()
    if not plot_df.empty:
        plot_df["short_method"] = plot_df["method_key"].str.slice(0, 80)

        for metric in ["macro_precision", "macro_recall", "macro_iou", "macro_boundary_f1"]:
            fig_h = max(5, 0.28 * len(plot_df))
            fig, ax = plt.subplots(figsize=(11, fig_h))
            y = np.arange(len(plot_df))
            ax.barh(y, plot_df[metric])
            ax.set_yticks(y)
            ax.set_yticklabels(plot_df["short_method"], fontsize=7)
            ax.invert_yaxis()
            ax.set_xlim(0, 1)
            ax.set_xlabel(metric)
            ax.set_title(f"Top {len(plot_df)} methods by macro IoU - {metric}")
            fig.tight_layout()
            fig.savefig(perf_dir / f"metric_bar_{metric}.png", dpi=160, bbox_inches="tight")
            plt.close(fig)

        heat_metrics = ["macro_precision", "macro_recall", "macro_iou", "macro_boundary_f1"]
        mat = plot_df[heat_metrics].to_numpy(dtype=float)
        fig_h = max(5, 0.32 * len(plot_df))
        fig, ax = plt.subplots(figsize=(8, fig_h))
        im = ax.imshow(mat, vmin=0, vmax=1, interpolation="nearest")
        ax.set_xticks(np.arange(len(heat_metrics)))
        ax.set_xticklabels(heat_metrics, rotation=30, ha="right")
        ax.set_yticks(np.arange(len(plot_df)))
        ax.set_yticklabels(plot_df["short_method"], fontsize=7)
        ax.set_title(f"Top {len(plot_df)} method metrics")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(perf_dir / "method_metric_heatmap.png", dpi=160, bbox_inches="tight")
        plt.close(fig)

    return macro_by_method


def evaluate(args: argparse.Namespace) -> None:
    root = Path(args.root).expanduser().resolve()
    out_dir = Path(args.out_dir)
    perf_dir = Path(args.perf_dir)

    if not out_dir.is_absolute():
        out_dir = root / out_dir
    if not perf_dir.is_absolute():
        perf_dir = root / perf_dir

    out_dir.mkdir(parents=True, exist_ok=True)
    perf_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "visuals").mkdir(parents=True, exist_ok=True)
    (perf_dir / "confusion_matrices").mkdir(parents=True, exist_ok=True)

    predictions = discover_predictions(root, args.gt_folder)
    if not predictions:
        raise SystemExit(
            "No prediction masks found. Check that files are under ImgXX/<feature>/<KMeans or Mean Shift>/ "
            "and filenames start with 'Post Processed' or 'Postprocessed'."
        )

    per_image_rows: List[Dict[str, object]] = []
    per_class_rows_all: List[Dict[str, object]] = []
    cm_rows_all: List[Dict[str, object]] = []
    label_match_rows_all: List[Dict[str, object]] = []
    qualitative_rows: List[Dict[str, object]] = []

    gt_cache: Dict[Tuple[str, Optional[str], Optional[str], str, bool], GTData] = {}

    for i, item in enumerate(predictions, start=1):
        print(f"[{i}/{len(predictions)}] {item.rel_path}")

        gt_cache_key = (
            item.image_id,
            str(item.gt_mask_path) if item.gt_mask_path else None,
            str(item.gt_json_path) if item.gt_json_path else None,
            args.gt_source,
            bool(args.name_gt_classes_from_json),
        )
        if gt_cache_key not in gt_cache:
            gt_cache[gt_cache_key] = load_gt_data(
                image_id=item.image_id,
                gt_mask_path=item.gt_mask_path,
                gt_json_path=item.gt_json_path,
                gt_source=args.gt_source,
                name_gt_classes_from_json=args.name_gt_classes_from_json,
            )

        gt_data = gt_cache[gt_cache_key]
        gt_labels = gt_data.labels
        gt_names = gt_data.class_names
        gt_raw = gt_data.raw_rgb

        pred_labels, pred_names, pred_raw = load_mask_as_labels(item.pred_path)
        pred_labels = resize_labels_nearest(pred_labels, gt_labels.shape)
        pred_raw = resize_rgb_nearest(pred_raw, gt_labels.shape)

        # Compact after resizing, because resizing can preserve sparse old ids.
        pred_labels, pred_old_ids = compact_labels(pred_labels)
        pred_names_resized = [
            pred_names[old_id] if 0 <= old_id < len(pred_names) else str(old_id)
            for old_id in pred_old_ids
        ]

        remapped_pred, matched_cm, column_names, match_rows = match_prediction_labels(
            gt=gt_labels,
            pred=pred_labels,
            gt_names=gt_names,
            pred_names=pred_names_resized,
            matching=args.matching,
        )

        summary, per_class_rows = compute_metrics(
            gt=gt_labels,
            remapped_pred=remapped_pred,
            matched_cm=matched_cm,
            gt_names=gt_names,
            tolerance=args.boundary_tolerance,
            include_background=args.include_background,
        )

        item_safe = safe_name(f"{item.image_id}__{item.feature_set}__{item.method_dir}__{item.prediction_name}")
        visual_path = out_dir / "visuals" / f"{item_safe}.png"
        cm_plot_path = perf_dir / "confusion_matrices" / f"{item_safe}.png"

        base_row = {
            "image_id": item.image_id,
            "feature_set": item.feature_set,
            "method_dir": item.method_dir,
            "algorithm": item.algorithm,
            "prediction_name": item.prediction_name,
            "method_key": item.method_key,
            "gt_source_type": gt_data.source_type,
            "gt_source_path": str(gt_data.source_path),
            "gt_mask_path": str(item.gt_mask_path) if item.gt_mask_path else "",
            "gt_json_path": str(item.gt_json_path) if item.gt_json_path else "",
            "prediction_path": str(item.pred_path),
            "prediction_rel_path": item.rel_path,
            "matching": args.matching,
            "gt_shape": f"{gt_labels.shape[0]}x{gt_labels.shape[1]}",
            "n_gt_labels": len(gt_names),
            "n_pred_labels": len(pred_names_resized),
            "boundary_tolerance_px": args.boundary_tolerance,
            "include_background_in_macro": args.include_background,
            "visual_result_path": str(visual_path),
            "confusion_matrix_plot_path": str(cm_plot_path),
        }

        per_image_row = {**base_row, **summary}
        per_image_rows.append(per_image_row)

        for row in per_class_rows:
            per_class_rows_all.append({**base_row, **row})

        for row in match_rows:
            label_match_rows_all.append({**base_row, **row})

        cm_norm = row_normalized(matched_cm)
        for r, gt_name in enumerate(gt_names):
            for c, pred_name in enumerate(column_names):
                cm_rows_all.append(
                    {
                        **base_row,
                        "gt_label_id": r,
                        "gt_label_name": gt_name,
                        "pred_label_after_matching_id": c,
                        "pred_label_after_matching_name": pred_name,
                        "raw_count": int(matched_cm[r, c]),
                        "row_normalized_value": float(cm_norm[r, c]),
                    }
                )

        qualitative_rows.append(
            {
                **base_row,
                "qualitative_panel": str(visual_path),
                "gt_visual": "panel column: GT raw / GT labels",
                "prediction_visual": "panel column: Prediction raw / Matched prediction",
                "error_visual": "panel column: Error pixels",
                **summary,
            }
        )

        save_visual_panel(
            out_path=visual_path,
            image_id=item.image_id,
            method_key=item.method_key,
            gt_raw=gt_raw,
            pred_raw=pred_raw,
            gt_labels=gt_labels,
            remapped_pred=remapped_pred,
        )
        save_confusion_heatmap(
            out_path=cm_plot_path,
            cm_norm=cm_norm,
            row_names=gt_names,
            col_names=column_names,
            title=f"{item.image_id} | {item.method_key}",
        )

    per_image_df = pd.DataFrame(per_image_rows)
    per_class_df = pd.DataFrame(per_class_rows_all)
    cm_df = pd.DataFrame(cm_rows_all)
    label_match_df = pd.DataFrame(label_match_rows_all)
    qualitative_df = pd.DataFrame(qualitative_rows)

    per_image_df.to_csv(out_dir / "per_image_results.csv", index=False)
    per_class_df.to_csv(out_dir / "per_class_results.csv", index=False)
    cm_df.to_csv(out_dir / "confusion_matrices_long.csv", index=False)
    label_match_df.to_csv(out_dir / "label_matches.csv", index=False)
    qualitative_df.to_csv(out_dir / "qualitative_results.csv", index=False)

    # Also save main quantitative files in performance/ for convenience.
    per_image_df.to_csv(perf_dir / "per_image_results.csv", index=False)
    per_class_df.to_csv(perf_dir / "per_class_results.csv", index=False)

    macro_df = plot_performance_summaries(per_image_df, perf_dir, args.top_n)

    print("\nDone.")
    print(f"Evaluated prediction masks: {len(per_image_df)}")
    print(f"Output CSVs and qualitative panels: {out_dir}")
    print(f"Performance summaries and plots: {perf_dir}")
    print("\nTop methods by macro IoU:")
    cols = ["method_key", "n_images", "macro_precision", "macro_recall", "macro_iou", "macro_boundary_f1"]
    print(macro_df[cols].head(min(10, len(macro_df))).to_string(index=False))


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare generated segmentation masks with GT masks using label matching."
    )
    parser.add_argument(
        "--root",
        required=True,
        help="Root folder containing GT folder and Img01, Img02, ... folders. Example: E:\\",
    )
    parser.add_argument(
        "--gt-folder",
        default="GT Ehsanul Karim",
        help="Ground-truth folder name under root. Default: GT Ehsanul Karim",
    )
    parser.add_argument(
        "--gt-source",
        choices=["mask", "json", "auto"],
        default="mask",
        help=(
            "GT source. 'mask' uses mask.png. 'json' rasterizes ImgXX.json. "
            "'auto' uses mask.png if available, otherwise JSON. Default: mask"
        ),
    )
    parser.add_argument(
        "--name-gt-classes-from-json",
        action="store_true",
        help=(
            "When --gt-source mask, read ImgXX.json and infer class names by overlap "
            "with mask.png labels. Useful for labels like Starfish / Stone."
        ),
    )
    parser.add_argument(
        "--out-dir",
        default="output",
        help="Directory for qualitative panels and detailed CSVs. Relative paths are created under root.",
    )
    parser.add_argument(
        "--perf-dir",
        default="performance",
        help="Directory for performance plots and summary CSVs. Relative paths are created under root.",
    )
    parser.add_argument(
        "--matching",
        choices=["hungarian", "many_to_one"],
        default="hungarian",
        help=(
            "Label matching strategy. Use hungarian for one-to-one label matching. "
            "Use many_to_one if generated masks are over-segmented and several predicted labels may correspond to one GT label."
        ),
    )
    parser.add_argument(
        "--boundary-tolerance",
        type=int,
        default=2,
        help="Boundary F1 tolerance in pixels. Default: 2",
    )
    parser.add_argument(
        "--include-background",
        action="store_true",
        help=(
            "Include class 0/background in macro Precision, Recall, IoU, and BF-score. "
            "By default, class 0 is excluded from macro averages but still appears in per-class CSV."
        ),
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=30,
        help="Number of top methods to show in summary charts. Default: 30",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    evaluate(parse_args(sys.argv[1:]))
