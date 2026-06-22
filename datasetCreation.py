import json
from pathlib import Path

import cv2
import numpy as np


json_path = Path("Img09.json")
image_path = Path("Img09.png")

output_dir = Path("outputs")
output_dir.mkdir(exist_ok=True)

with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

image = cv2.imread(str(image_path))
if image is None:
    raise FileNotFoundError(f"Could not read image: {image_path}")

h, w = image.shape[:2]

mask = np.full((h, w, 3), (255, 255, 0), dtype=np.uint8)  # cyan background
contour = np.zeros((h, w, 3), dtype=np.uint8)              # black background
overlay = image.copy()

# OpenCV uses BGR color order
label_colors = {
    "Bin 1": (0, 220, 220),       # yellow
    "Bin 2": (0, 255, 0),         # green
    "Bin 3": (255, 0, 0),         # blue
    "Bin 4": (0, 0, 255),         # red
    "Bin 5": (255, 255, 0),       # cyan
}

for shape in data["shapes"]:
    if shape.get("shape_type", "polygon") != "polygon":
        continue

    label = shape["label"]
    points = np.array(shape["points"], dtype=np.float32)
    points = np.round(points).astype(np.int32)

    pts = points.reshape((-1, 1, 2))

    fill_color = label_colors.get(label, (200, 200, 200))

    # 1. Filled mask
    cv2.fillPoly(mask, [pts], fill_color)

    # 2. Contour image
    cv2.polylines(
        contour,
        [pts],
        isClosed=True,
        color=(255, 255, 255),
        thickness=2
    )

    # 3. Contour overlay on original image
    cv2.polylines(
        overlay,
        [pts],
        isClosed=True,
        color=(0, 255, 0),
        thickness=2
    )

cv2.imwrite(str(output_dir / "mask.png"), mask)
cv2.imwrite(str(output_dir / "contour.png"), contour)
cv2.imwrite(str(output_dir / "contour_overlay.png"), overlay)