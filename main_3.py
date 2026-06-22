"""
Optimized image segmentation script.

Main changes from the original version:
- Uses scikit-learn / OpenCV library implementations instead of hand-written
  clustering loops.
- Keeps feature matrices as contiguous float32 NumPy arrays.
- Computes color, XY, and texture features lazily and caches them so texture
  filters are not recomputed for every feature-space combination.
- Uses vectorized nearest-center assignment for sampled Mean Shift results.
- Disables cv2.imshow by default for faster non-interactive runs.

Folder convention is unchanged:
    outputs/{ImgName}/{ImgName}.jpg or .png

Results are saved under:
    outputs/{ImgName}/{feature_option}/{algorithm}/...
"""

from __future__ import annotations

import os
from enum import Enum
from typing import Iterable, Sequence

import cv2
import numpy as np

try:
    from sklearn.cluster import MeanShift, MiniBatchKMeans
    from sklearn.metrics import pairwise_distances_argmin
except ImportError as exc:  # pragma: no cover - user environment guard
    MeanShift = None
    MiniBatchKMeans = None
    pairwise_distances_argmin = None
    SKLEARN_IMPORT_ERROR = exc
else:
    SKLEARN_IMPORT_ERROR = None


# Set to True only when you want GUI preview windows.
SHOW_WINDOWS = False

# Global names are preserved so output folder behavior remains familiar.
image_name_saved: str = ""
selectedoptionSaved = None


class SelectedOption(Enum):
    xy = 1
    rgb = 2
    hsv = 3
    lab = 4
    texture = 5
    xy_rgb = 6
    xy_lab = 7
    xy_hsv = 8
    xy_texture = 9
    rgb_texture = 10
    hsv_texture = 11
    lab_texture = 12
    xy_rgb_texture = 13
    xy_lab_texture = 14
    xy_hsv_texture = 15


FEATURE_SPACE_OPTIONS = {
    "1": (SelectedOption.xy, ("xy",)),
    "2": (SelectedOption.rgb, ("rgb",)),
    "3": (SelectedOption.hsv, ("hsv",)),
    "4": (SelectedOption.lab, ("lab",)),
    "5": (SelectedOption.texture, ("texture",)),
    "6": (SelectedOption.xy_rgb, ("xy", "rgb")),
    "7": (SelectedOption.xy_lab, ("xy", "lab")),
    "8": (SelectedOption.xy_hsv, ("xy", "hsv")),
    "9": (SelectedOption.xy_texture, ("xy", "texture")),
    "10": (SelectedOption.rgb_texture, ("rgb", "texture")),
    "11": (SelectedOption.hsv_texture, ("hsv", "texture")),
    "12": (SelectedOption.lab_texture, ("lab", "texture")),
    "13": (SelectedOption.xy_rgb_texture, ("xy", "rgb", "texture")),
    "14": (SelectedOption.xy_lab_texture, ("xy", "lab", "texture")),
    "15": (SelectedOption.xy_hsv_texture, ("xy", "hsv", "texture")),
}


# ----------------------------- I/O helpers -----------------------------


def load_image() -> np.ndarray | None:
    img_name = input("Enter the image name (e.g., Img01, Img02): ").strip()

    global image_name_saved
    image_name_saved = img_name

    img_path_jpg = os.path.join("outputs", img_name, f"{img_name}.jpg")
    img_path_png = os.path.join("outputs", img_name, f"{img_name}.png")

    if os.path.exists(img_path_jpg):
        img = cv2.imread(img_path_jpg, cv2.IMREAD_COLOR)
        print(f"Loaded image from {img_path_jpg}")
        return img

    if os.path.exists(img_path_png):
        img = cv2.imread(img_path_png, cv2.IMREAD_COLOR)
        print(f"Loaded image from {img_path_png}")
        return img

    print("Image not found. Please check the name and try again.")
    return None


def save_image(image: np.ndarray, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ok = cv2.imwrite(path, image)
    if not ok:
        raise IOError(f"Could not save image to {path}")


def maybe_show_image(window_name: str, image: np.ndarray) -> None:
    if not SHOW_WINDOWS:
        return
    cv2.imshow(window_name, image)
    cv2.waitKey(1)


# -------------------------- Feature extraction --------------------------


def generate_gabor_kernels(
    ksize: int,
    wavelengths: Sequence[float],
    orientations: Sequence[float],
    phase_offsets: Sequence[float],
    std_dev: float,
    spatial_aspect_ratio: float,
) -> list[np.ndarray]:
    """Create OpenCV Gabor kernels as float32 arrays."""
    kernels: list[np.ndarray] = []
    for theta in orientations:
        for phi in phase_offsets:
            for wavelength in wavelengths:
                kernel = cv2.getGaborKernel(
                    (ksize, ksize),
                    std_dev,
                    theta,
                    wavelength,
                    spatial_aspect_ratio,
                    phi,
                    ktype=cv2.CV_32F,
                )
                kernels.append(kernel)
    return kernels


def normalize_feature_matrix(features_hwf: np.ndarray) -> np.ndarray:
    """Flatten HxWxF features to NxF and standardize each feature column."""
    X = features_hwf.reshape(-1, features_hwf.shape[-1]).astype(np.float32, copy=False)

    # Accumulate in float64 for stable statistics, return float32 for speed/memory.
    mean = X.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = X.std(axis=0, dtype=np.float64).astype(np.float32)

    return ((X - mean) / (std + 1e-6)).astype(np.float32, copy=False)


class FeatureExtractor:
    """
    Lazy, cached feature extractor.

    This prevents expensive repeated work in "run all combinations" mode,
    especially Gabor texture filtering.
    """

    def __init__(self, image: np.ndarray):
        self.image = image
        self.h, self.w = image.shape[:2]
        self._cache: dict[str, list[np.ndarray]] = {}

    def get(self, part: str) -> list[np.ndarray]:
        if part == "xy":
            return self.xy()
        if part == "rgb":
            return self.rgb()
        if part == "hsv":
            return self.hsv()
        if part == "lab":
            return self.lab()
        if part == "texture":
            return self.texture()
        raise ValueError(f"Unknown feature part: {part}")

    def xy(self) -> list[np.ndarray]:
        cached = self._cache.get("xy")
        if cached is not None:
            return cached

        x_coords, y_coords = np.meshgrid(
            np.arange(self.w, dtype=np.float32),
            np.arange(self.h, dtype=np.float32),
        )
        x_coords /= max(self.w - 1, 1)
        y_coords /= max(self.h - 1, 1)

        self._cache["xy"] = [x_coords, y_coords]
        return self._cache["xy"]

    def rgb(self) -> list[np.ndarray]:
        cached = self._cache.get("rgb")
        if cached is not None:
            return cached

        rgb_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB).astype(np.float32)
        self._cache["rgb"] = list(cv2.split(rgb_image))
        return self._cache["rgb"]

    def hsv(self) -> list[np.ndarray]:
        cached = self._cache.get("hsv")
        if cached is not None:
            return cached

        hsv_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV).astype(np.float32)
        self._cache["hsv"] = list(cv2.split(hsv_image))
        return self._cache["hsv"]

    def lab(self) -> list[np.ndarray]:
        cached = self._cache.get("lab")
        if cached is not None:
            return cached

        lab_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2LAB).astype(np.float32)
        self._cache["lab"] = list(cv2.split(lab_image))
        return self._cache["lab"]

    def texture(self) -> list[np.ndarray]:
        cached = self._cache.get("texture")
        if cached is not None:
            return cached

        gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY).astype(np.float32)

        kernels = generate_gabor_kernels(
            ksize=31,
            wavelengths=(4, 8, 16, 32),
            orientations=(0, np.pi / 4, np.pi / 2, 3 * np.pi / 4),
            phase_offsets=(0, 0.8),
            std_dev=4.0,
            spatial_aspect_ratio=0.1,
        )

        texture_features = [
            cv2.filter2D(gray_image, cv2.CV_32F, kernel, borderType=cv2.BORDER_REFLECT)
            for kernel in kernels
        ]

        self._cache["texture"] = texture_features
        return self._cache["texture"]

    def create_feature_space(self, parts: Iterable[str]) -> tuple[np.ndarray, np.ndarray]:
        feature_planes: list[np.ndarray] = []
        for part in parts:
            feature_planes.extend(self.get(part))

        features_hwf = np.dstack(feature_planes).astype(np.float32, copy=False)
        X = normalize_feature_matrix(features_hwf)
        return features_hwf, X


# ------------------------- Visualization helpers ------------------------


def labels_to_segmented_image(image: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    h, w = image.shape[:2]
    label_image = labels.reshape(h, w).astype(np.int32, copy=False)

    # Keep labels as float32 while normalizing to avoid uint8 overflow when labels > 255.
    label_display = cv2.normalize(
        label_image.astype(np.float32), None, 0, 255, cv2.NORM_MINMAX
    ).astype(np.uint8)
    label_display = cv2.applyColorMap(label_display, cv2.COLORMAP_JET)
    return label_image, label_display


def output_dir(algorithm_name: str) -> str:
    if selectedoptionSaved is None:
        feature_name = "unknown_feature_space"
    else:
        feature_name = selectedoptionSaved.name

    path = os.path.join("outputs", image_name_saved, feature_name, algorithm_name)
    os.makedirs(path, exist_ok=True)
    return path


# ---------------------------- Postprocess -------------------------------


def postprocess_labels(label_image: np.ndarray, min_area: int = 100) -> np.ndarray:
    """Remove small connected components by assigning them to surrounding labels."""
    cleaned = label_image.copy()
    unique_labels = np.unique(label_image)

    for label in unique_labels:
        mask = (label_image == label).astype(np.uint8)
        num_components, components, stats, _ = cv2.connectedComponentsWithStats(
            mask, connectivity=8
        )

        for component_id in range(1, num_components):
            area = stats[component_id, cv2.CC_STAT_AREA]
            if area >= min_area:
                continue

            component_mask = components == component_id
            dilated = cv2.dilate(
                component_mask.astype(np.uint8),
                np.ones((3, 3), np.uint8),
                iterations=1,
            )
            border_mask = (dilated == 1) & (~component_mask)
            surrounding_labels = label_image[border_mask]

            if surrounding_labels.size > 0:
                # Works for non-negative integer labels.
                new_label = np.bincount(surrounding_labels.astype(np.int32)).argmax()
                cleaned[component_mask] = new_label

    return cleaned


# ------------------------ Library-based clustering ----------------------


def apply_k_means_segmentation(
    image: np.ndarray,
    X: np.ndarray,
    k: int = 5,
    batch_size: int = 8192,
    max_iter: int = 100,
    random_state: int = 42,
) -> np.ndarray:
    """
    Fast KMeans using scikit-learn MiniBatchKMeans.

    This replaces the old Python loop that repeatedly computed distances,
    assigned clusters, and recomputed centroids manually.
    """
    if MiniBatchKMeans is None:
        raise ImportError(
            "scikit-learn is required for MiniBatchKMeans. "
            f"Original error: {SKLEARN_IMPORT_ERROR}"
        )

    X = np.ascontiguousarray(X, dtype=np.float32)

    print(f"Running library MiniBatchKMeans: k={k}, batch_size={batch_size}")
    model = MiniBatchKMeans(
        n_clusters=k,
        init="k-means++",
        max_iter=max_iter,
        batch_size=batch_size,
        n_init="auto",
        random_state=random_state,
        compute_labels=True,
    )
    labels = model.fit_predict(X).astype(np.int32, copy=False)

    label_image, label_display = labels_to_segmented_image(image, labels)
    maybe_show_image("Segmented Image (MiniBatchKMeans)", label_display)

    img_path = output_dir("KMeans")
    save_image(label_display, os.path.join(img_path, f"{image_name_saved}-segment of {k}.png"))

    postprocessed_labels = postprocess_labels(label_image)
    _, postprocessed_display = labels_to_segmented_image(
        image, postprocessed_labels.reshape(-1)
    )
    maybe_show_image("Postprocessed Segmented Image (MiniBatchKMeans)", postprocessed_display)
    save_image(
        postprocessed_display,
        os.path.join(img_path, f"Postprocessed {image_name_saved}-segment of {k}.png"),
    )

    return labels


def sample_pixels(
    X: np.ndarray,
    sample_ratio: float = 0.07,
    max_samples: int = 12000,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Randomly sample pixels with an upper bound to keep Mean Shift practical."""
    rng = np.random.default_rng(seed)
    n_pixels = X.shape[0]
    sample_size = max(1, int(n_pixels * sample_ratio))
    sample_size = min(sample_size, max_samples, n_pixels)

    sample_indices = rng.choice(n_pixels, size=sample_size, replace=False)
    sample_X = np.ascontiguousarray(X[sample_indices], dtype=np.float32)
    return sample_indices, sample_X


def assign_all_pixels_to_nearest_modes(
    X: np.ndarray,
    cluster_centers: np.ndarray,
) -> np.ndarray:
    """Use sklearn's optimized pairwise assignment instead of manual loops."""
    if pairwise_distances_argmin is None:
        raise ImportError(
            "scikit-learn is required for optimized nearest-center assignment. "
            f"Original error: {SKLEARN_IMPORT_ERROR}"
        )

    X = np.ascontiguousarray(X, dtype=np.float32)
    centers = np.ascontiguousarray(cluster_centers, dtype=np.float32)
    return pairwise_distances_argmin(X, centers, metric="euclidean").astype(np.int32)


def mean_shift_segmentation(
    image: np.ndarray,
    X: np.ndarray,
    bandwidths: Sequence[float],
    sample_ratio: float = 0.07,
    max_samples: int = 12000,
    max_iter: int = 100,
    seed: int = 42,
) -> None:
    """
    Sampled Mean Shift using scikit-learn's optimized implementation.

    The model is fitted on sampled pixels, then every original pixel is assigned
    to the nearest discovered mode. This keeps the output image full-resolution
    but avoids shifting every pixel manually.
    """
    if MeanShift is None:
        raise ImportError(
            "scikit-learn is required for MeanShift. "
            f"Original error: {SKLEARN_IMPORT_ERROR}"
        )

    h, w = image.shape[:2]
    n_pixels = h * w
    if X.shape[0] != n_pixels:
        raise ValueError(
            f"X has {X.shape[0]} rows, but image has {n_pixels} pixels. "
            "X must have one row per pixel."
        )

    X = np.ascontiguousarray(X, dtype=np.float32)
    _, sample_X = sample_pixels(X, sample_ratio=sample_ratio, max_samples=max_samples, seed=seed)
    print(
        f"Using {sample_X.shape[0]} sampled pixels out of {X.shape[0]} total pixels "
        f"({100.0 * sample_X.shape[0] / X.shape[0]:.2f}%)."
    )

    img_path = output_dir("Sampled Mean Shift")

    for bandwidth in bandwidths:
        print("=" * 60)
        print(f"Running library MeanShift: bandwidth={bandwidth}")

        model = MeanShift(
            bandwidth=bandwidth,
            bin_seeding=True,
            cluster_all=True,
            n_jobs=1,
            max_iter=max_iter,
        )
        try:
            model.fit(sample_X)
            cluster_centers = np.ascontiguousarray(model.cluster_centers_, dtype=np.float32)

            print(f"Found {cluster_centers.shape[0]} modes from sampled pixels.")
            print("Assigning every original pixel to the nearest mode...")
            labels = assign_all_pixels_to_nearest_modes(X, cluster_centers)

            label_image, segmented_image = labels_to_segmented_image(image, labels)
            maybe_show_image(f"Sampled Mean Shift - h={bandwidth}", segmented_image)
            save_image(segmented_image, os.path.join(img_path, f"{image_name_saved}-meanshift-{bandwidth}.png"))

            postprocessed_labels = postprocess_labels(label_image)
            _, postprocessed_image = labels_to_segmented_image(
                image, postprocessed_labels.reshape(-1)
            )
            maybe_show_image(f"Sampled Mean Shift Post Processed - h={bandwidth}", postprocessed_image)
            save_image(
                postprocessed_image,
                os.path.join(img_path, f"Post Processed {image_name_saved}-meanshift-{bandwidth}.png"),
            ) 
        except:
            print("Do it manually by setting some other bandwidth")


# ------------------------------ CLI flow --------------------------------


def print_feature_space_menu() -> None:
    print("Select a feature space creation method:")
    for option_number, (selected_option, _) in FEATURE_SPACE_OPTIONS.items():
        print(f"{option_number}. {selected_option.name}")
    print("0. Run all combinations")


def select_feature_space(
    extractor: FeatureExtractor,
    option: str,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    global selectedoptionSaved

    if option not in FEATURE_SPACE_OPTIONS:
        return None, None

    selectedoptionSaved, parts = FEATURE_SPACE_OPTIONS[option]
    print(f"Using feature combination: {selectedoptionSaved.name}")
    return extractor.create_feature_space(parts)


def ask_k(default: int = 5) -> int:
    text = input(f"Enter the number of segments k [{default}]: ").strip()
    if not text:
        return default
    return int(text)


def ask_bandwidths(defaults: Sequence[float] = (0.5, 1.5)) -> list[float]:
    print(
        "Enter bandwidth values separated by spaces "
        f"or press Enter for defaults {list(defaults)}:"
    )
    text = input("Bandwidths: ").strip()
    if not text:
        return list(defaults)
    return [float(x) for x in text.split()]


def run_clustering_for_current_feature_space(
    image: np.ndarray,
    X: np.ndarray,
    clustering_option: str,
) -> None:
    if clustering_option == "1":
        apply_k_means_segmentation(image, X, k=ask_k())
    elif clustering_option == "2":
        mean_shift_segmentation(image, X, bandwidths=ask_bandwidths())
    elif clustering_option == "3":
        apply_k_means_segmentation(image, X, k=ask_k())
        mean_shift_segmentation(image, X, bandwidths=ask_bandwidths())
    else:
        raise ValueError("Invalid clustering option selected.")


def do_all(image: np.ndarray, extractor: FeatureExtractor) -> None:
    """Run all 15 feature-space combinations with cached feature extraction."""
    global selectedoptionSaved

    for option_number, (selected_option, parts) in FEATURE_SPACE_OPTIONS.items():
        selectedoptionSaved = selected_option
        print("=" * 80)
        print(f"Running combination {option_number}: {selectedoptionSaved.name}")

        _, X = extractor.create_feature_space(parts)
        apply_k_means_segmentation(image, X, k=5)
        mean_shift_segmentation(image, X, bandwidths=(0.8, 1.5))


def main() -> None:
    image = load_image()
    if image is None:
        return

    extractor = FeatureExtractor(image)

    print_feature_space_menu()
    option = input("Enter the option number: ").strip()

    if option == "0":
        do_all(image, extractor)
        return

    _, X = select_feature_space(extractor, option)
    if X is None:
        print("Invalid option selected. Running all combinations instead.")
        do_all(image, extractor)
        return

    print("Select a clustering algorithm:")
    print("1. KMeans (library MiniBatchKMeans)")
    print("2. Mean Shift (library sklearn MeanShift)")
    print("3. Both KMeans and Mean Shift")

    clustering_option = input("Enter the option number: ").strip()
    run_clustering_for_current_feature_space(image, X, clustering_option)

    if SHOW_WINDOWS:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
