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

When a texture feature space is selected, the texture extractor is now chosen
by the user:
    1. Gabor filter bank
    2. Leung-Malik filter bank
"""

from __future__ import annotations

import os
from enum import Enum
from typing import Iterable, Sequence

import cv2
import numpy as np


try:
    from sklearn.cluster import MeanShift, MiniBatchKMeans
    from sklearn.decomposition import PCA
    from sklearn.metrics import pairwise_distances_argmin
except ImportError as exc:  # pragma: no cover - user environment guard
    MeanShift = None
    MiniBatchKMeans = None
    PCA = None
    pairwise_distances_argmin = None
    SKLEARN_IMPORT_ERROR = exc
else:
    SKLEARN_IMPORT_ERROR = None


# Set to True only when you want GUI preview windows.
SHOW_WINDOWS = False

# Shows the Gabor filter-bank activation heat map and pauses on it with cv2.waitKey.
# Set to False if you run this script in a headless/non-GUI environment.
SHOW_GABOR_FILTER_BANK_HEATMAP = True

# Global names are preserved so output folder behavior remains familiar.
image_name_saved: str = ""
selectedoptionSaved = None
selected_texture_method_saved: str = "gabor"


TEXTURE_EXTRACTION_OPTIONS = {
    "1": {
        "key": "gabor",
        "name": "Gabor filter bank",
        "description": "Current method: multi-scale, multi-orientation Gabor responses.",
    },
    "2": {
        "key": "leung_malik",
        "name": "Leung-Malik filter bank",
        "description": "48-filter LM bank: oriented Gaussian derivatives, LoG, and Gaussian filters.",
    },
}


def texture_method_display_name(method_key: str) -> str:
    """Return a human-readable name for a texture method key."""
    for config in TEXTURE_EXTRACTION_OPTIONS.values():
        if config["key"] == method_key:
            return str(config["name"])
    return method_key


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


MEAN_SHIFT_PROCESSING_OPTIONS = {
    "1": {
        "name": "Sampled Mean Shift",
        "description": "Current behavior: sample pixels, fit Mean Shift, assign all pixels to nearest mode.",
        "scale": 1.0,
        "use_sampling": True,
        "use_pca": False,
    },
    "2": {
        "name": "Quarter Resize Mean Shift",
        "description": "Resize width and height to one fourth, then run Mean Shift without pixel sampling.",
        "scale": 0.25,
        "use_sampling": False,
        "use_pca": False,
    },
    "4": {
        "name": "Half Resize PCA 95 Mean Shift",
        "description": "Resize width and height to one half, then apply PCA retaining 95% variance before Mean Shift.",
        "scale": 0.5,
        "use_sampling": False,
        "use_pca": True,
    },
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


def show_image_with_wait(window_name: str, image: np.ndarray, wait_ms: int = 0) -> None:
    """Show an image and pause with cv2.waitKey when GUI display is enabled."""
    cv2.imshow(window_name, image)
    cv2.waitKey(wait_ms)


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


def normalize_filter_kernel(kernel: np.ndarray, zero_mean: bool = True) -> np.ndarray:
    """Normalize a filter kernel so response scales are easier to compare."""
    kernel = kernel.astype(np.float32, copy=True)
    if zero_mean:
        kernel -= np.mean(kernel, dtype=np.float64).astype(np.float32)

    norm = float(np.sum(np.abs(kernel), dtype=np.float64))
    if norm > 1e-8:
        kernel /= norm
    return kernel.astype(np.float32, copy=False)


def gaussian_grid(sigma: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return X/Y coordinates and a Gaussian envelope for the requested sigma."""
    radius = max(1, int(np.ceil(3.0 * sigma)))
    coords = np.arange(-radius, radius + 1, dtype=np.float32)
    x, y = np.meshgrid(coords, coords)
    sigma2 = float(sigma * sigma)
    gaussian = np.exp(-(x * x + y * y) / (2.0 * sigma2)).astype(np.float32)
    gaussian /= np.sum(gaussian, dtype=np.float64).astype(np.float32)
    return x, y, gaussian


def rotate_kernel(kernel: np.ndarray, theta: float) -> np.ndarray:
    """Rotate a kernel around its center without changing its size."""
    h, w = kernel.shape[:2]
    center = ((w - 1) / 2.0, (h - 1) / 2.0)
    matrix = cv2.getRotationMatrix2D(center, np.degrees(theta), 1.0)
    rotated = cv2.warpAffine(
        kernel,
        matrix,
        (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return rotated.astype(np.float32, copy=False)


def generate_leung_malik_kernels() -> list[np.ndarray]:
    """
    Create a Leung-Malik-style 48-filter texture bank.

    The bank contains:
    - 36 oriented filters: first and second Gaussian derivatives at 3 scales
      and 6 orientations.
    - 8 isotropic Laplacian-of-Gaussian filters.
    - 4 isotropic Gaussian smoothing filters.
    """
    kernels: list[np.ndarray] = []
    orientations = tuple(i * np.pi / 6.0 for i in range(6))
    derivative_sigmas = (1.0, np.sqrt(2.0), 2.0)
    gaussian_sigmas = (1.0, np.sqrt(2.0), 2.0, 2.0 * np.sqrt(2.0))

    for sigma in derivative_sigmas:
        x, _, gaussian = gaussian_grid(float(sigma))
        sigma2 = float(sigma * sigma)
        sigma4 = sigma2 * sigma2

        first_derivative = (-x / sigma2) * gaussian
        second_derivative = ((x * x - sigma2) / sigma4) * gaussian

        for theta in orientations:
            kernels.append(normalize_filter_kernel(rotate_kernel(first_derivative, theta)))
            kernels.append(normalize_filter_kernel(rotate_kernel(second_derivative, theta)))

    for sigma in gaussian_sigmas:
        x, y, gaussian = gaussian_grid(float(sigma))
        sigma2 = float(sigma * sigma)
        sigma4 = sigma2 * sigma2
        log_kernel = ((x * x + y * y - 2.0 * sigma2) / sigma4) * gaussian
        kernels.append(normalize_filter_kernel(log_kernel))

    for sigma in gaussian_sigmas:
        x, y, gaussian = gaussian_grid(float(sigma * 3.0))
        sigma2 = float((sigma * 3.0) * (sigma * 3.0))
        sigma4 = sigma2 * sigma2
        log_kernel = ((x * x + y * y - 2.0 * sigma2) / sigma4) * gaussian
        kernels.append(normalize_filter_kernel(log_kernel))

    for sigma in gaussian_sigmas:
        _, _, gaussian = gaussian_grid(float(sigma))
        kernels.append(normalize_filter_kernel(gaussian, zero_mean=False))

    return kernels


def apply_filter_bank(
    gray_image: np.ndarray,
    kernels: Sequence[np.ndarray],
) -> list[np.ndarray]:
    """Filter the grayscale image with a bank of kernels."""
    features: list[np.ndarray] = []
    for kernel in kernels:
        response = cv2.filter2D(gray_image, cv2.CV_32F, kernel, borderType=cv2.BORDER_REFLECT)
        features.append(response.astype(np.float32, copy=False))
    return features


def create_filter_bank_activation_heatmap(
    responses: Sequence[np.ndarray],
    colormap: int = cv2.COLORMAP_JET,
) -> np.ndarray:
    """Create a normalized JET heat map from filter-bank response strengths."""
    if not responses:
        raise ValueError("At least one filter response is required to create a heat map.")

    # Absolute response strength shows where any filter in the bank activates.
    response_stack = np.stack(
        [np.abs(response).astype(np.float32, copy=False) for response in responses],
        axis=-1,
    )
    activation_map = np.max(response_stack, axis=-1)

    normalized_activation = cv2.normalize(
        activation_map, None, 0, 255, cv2.NORM_MINMAX
    ).astype(np.uint8)
    return cv2.applyColorMap(normalized_activation, colormap)


def save_and_show_gabor_filter_bank_heatmap(responses: Sequence[np.ndarray]) -> None:
    """Save and optionally show the Gabor filter-bank activation heat map."""
    heatmap = create_filter_bank_activation_heatmap(responses, colormap=cv2.COLORMAP_JET)

    output_root = os.path.join("outputs", image_name_saved) if image_name_saved else "."
    output_path = os.path.join(output_root, "0.Gabor filter bank.png")
    save_image(heatmap, output_path)
    print(f"Saved normalized Gabor filter-bank heat map to {output_path}")

    if SHOW_GABOR_FILTER_BANK_HEATMAP:
        show_image_with_wait("0.Gabor filter bank", heatmap, wait_ms=0)


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
    especially texture filtering.
    """

    def __init__(
        self,
        image: np.ndarray,
        texture_method: str = "gabor",
    ):
        if texture_method not in {config["key"] for config in TEXTURE_EXTRACTION_OPTIONS.values()}:
            raise ValueError(f"Unknown texture extraction method: {texture_method}")

        self.image = image
        self.h, self.w = image.shape[:2]
        self.texture_method = texture_method
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
        cache_key = f"texture:{self.texture_method}"
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY).astype(np.float32)

        if self.texture_method == "gabor":
            kernels = generate_gabor_kernels(
                ksize=31,
                wavelengths=(4, 8, 16, 32),
                orientations=(0, np.pi / 4, np.pi / 2, 3 * np.pi / 4),
                phase_offsets=(0, 0.8),
                std_dev=4.0,
                spatial_aspect_ratio=0.1,
            )
            texture_features = apply_filter_bank(gray_image, kernels)
            save_and_show_gabor_filter_bank_heatmap(texture_features)

        elif self.texture_method == "leung_malik":
            kernels = generate_leung_malik_kernels()
            texture_features = apply_filter_bank(gray_image, kernels)

        else:  # pragma: no cover - guarded by __init__ validation
            raise ValueError(f"Unknown texture extraction method: {self.texture_method}")

        print(
            f"Computed {len(texture_features)} texture features using "
            f"{texture_method_display_name(self.texture_method)}."
        )
        self._cache[cache_key] = texture_features
        return self._cache[cache_key]

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

    if "texture" in feature_name:
        feature_name = f"{feature_name}_{selected_texture_method_saved}"

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


def resize_image_for_processing(image: np.ndarray, scale: float) -> np.ndarray:
    """Resize an image for faster Mean Shift computation."""
    if scale <= 0:
        raise ValueError("Resize scale must be greater than 0.")

    if scale == 1.0:
        return image

    h, w = image.shape[:2]
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(image, (new_w, new_h), interpolation=interpolation)
    print(f"Resized image from {w}x{h} to {new_w}x{new_h} for Mean Shift.")
    return resized


def apply_pca_95_variance(
    X: np.ndarray,
    variance_to_keep: float = 0.95,
) -> np.ndarray:
    """Reduce feature dimensions with PCA while retaining the requested variance."""
    if PCA is None:
        raise ImportError(
            "scikit-learn is required for PCA. "
            f"Original error: {SKLEARN_IMPORT_ERROR}"
        )

    X = np.ascontiguousarray(X, dtype=np.float32)
    original_dim = X.shape[1]

    if original_dim <= 1:
        print("Skipping PCA because the feature matrix has only one dimension.")
        return X

    model = PCA(n_components=variance_to_keep, svd_solver="full")
    X_pca = model.fit_transform(X).astype(np.float32, copy=False)
    retained = float(np.sum(model.explained_variance_ratio_))

    print(
        f"PCA reduced feature dimensions from {original_dim} to {X_pca.shape[1]} "
        f"while retaining {retained:.2%} variance."
    )
    return np.ascontiguousarray(X_pca, dtype=np.float32)


def prepare_mean_shift_inputs(
    image: np.ndarray,
    X: np.ndarray,
    parts: Iterable[str],
    processing_mode: str,
) -> tuple[np.ndarray, np.ndarray, str, bool]:
    """Prepare image/features according to the selected Mean Shift processing mode."""
    if processing_mode not in MEAN_SHIFT_PROCESSING_OPTIONS:
        print("Invalid Mean Shift processing option selected. Using option 1.")
        processing_mode = "1"

    config = MEAN_SHIFT_PROCESSING_OPTIONS[processing_mode]
    algorithm_name = str(config["name"])
    use_sampling = bool(config["use_sampling"])
    scale = float(config["scale"])
    use_pca = bool(config["use_pca"])

    processed_image = image
    processed_X = X

    if scale != 1.0:
        processed_image = resize_image_for_processing(image, scale)
        resized_extractor = FeatureExtractor(
            processed_image,
            texture_method=selected_texture_method_saved,
        )
        _, processed_X = resized_extractor.create_feature_space(parts)

    if use_pca:
        processed_X = apply_pca_95_variance(processed_X, variance_to_keep=0.95)

    return processed_image, processed_X, algorithm_name, use_sampling


def mean_shift_segmentation(
    image: np.ndarray,
    X: np.ndarray,
    bandwidths: Sequence[float],
    sample_ratio: float = 0.07,
    max_samples: int = 12000,
    max_iter: int = 100,
    seed: int = 42,
    algorithm_name: str = "Sampled Mean Shift",
    use_sampling: bool = True,
) -> None:
    """
    Run Mean Shift using one of the selected computation strategies.

    If use_sampling=True, the model is fitted on sampled pixels and every pixel
    is assigned to the nearest discovered mode. If use_sampling=False, Mean Shift
    is fitted on every row in X, which is used by the resize/PCA modes.
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
            "X must have one row per pixel in the image passed to Mean Shift."
        )

    X = np.ascontiguousarray(X, dtype=np.float32)

    if use_sampling:
        _, fit_X = sample_pixels(
            X,
            sample_ratio=sample_ratio,
            max_samples=max_samples,
            seed=seed,
        )
        print(
            f"Using {fit_X.shape[0]} sampled pixels out of {X.shape[0]} total pixels "
            f"({100.0 * fit_X.shape[0] / X.shape[0]:.2f}%)."
        )
    else:
        fit_X = X
        print(f"Using all {fit_X.shape[0]} pixels for Mean Shift; no pixel sampling.")

    img_path = output_dir(algorithm_name)

    for bandwidth in bandwidths:
        print("=" * 60)
        print(f"Running library MeanShift: bandwidth={bandwidth}")
        print(f"Processing mode: {algorithm_name}")

        model = MeanShift(
            bandwidth=bandwidth,
            bin_seeding=True,
            cluster_all=True,
            n_jobs=1,
            max_iter=max_iter,
        )
        try:
            if use_sampling:
                model.fit(fit_X)
                cluster_centers = np.ascontiguousarray(
                    model.cluster_centers_, dtype=np.float32
                )

                print(f"Found {cluster_centers.shape[0]} modes from sampled pixels.")
                print("Assigning every processed pixel to the nearest mode...")
                labels = assign_all_pixels_to_nearest_modes(X, cluster_centers)
            else:
                labels = model.fit_predict(fit_X).astype(np.int32, copy=False)
                print(f"Found {model.cluster_centers_.shape[0]} modes.")

            label_image, segmented_image = labels_to_segmented_image(image, labels)
            maybe_show_image(f"{algorithm_name} - h={bandwidth}", segmented_image)
            save_image(
                segmented_image,
                os.path.join(img_path, f"{image_name_saved}-meanshift-{bandwidth}.png"),
            )

            postprocessed_labels = postprocess_labels(label_image)
            _, postprocessed_image = labels_to_segmented_image(
                image, postprocessed_labels.reshape(-1)
            )
            maybe_show_image(
                f"{algorithm_name} Post Processed - h={bandwidth}",
                postprocessed_image,
            )
            save_image(
                postprocessed_image,
                os.path.join(
                    img_path,
                    f"Post Processed {image_name_saved}-meanshift-{bandwidth}.png",
                ),
            )
        except Exception as exc:
            print(
                "Mean Shift failed for this bandwidth. "
                f"Try another bandwidth or a faster processing mode. Details: {exc}"
            )


# ------------------------------ CLI flow --------------------------------


def print_feature_space_menu() -> None:
    print("Select a feature space creation method:")
    for option_number, (selected_option, _) in FEATURE_SPACE_OPTIONS.items():
        print(f"{option_number}. {selected_option.name}")
    print("0. Run all combinations")


def print_mean_shift_processing_menu() -> None:
    print("Select a Mean Shift computation mode:")
    for option_number, config in MEAN_SHIFT_PROCESSING_OPTIONS.items():
        print(f"{option_number}. {config['description']}")


def print_texture_extraction_menu() -> None:
    print("Select a texture feature extraction method:")
    for option_number, config in TEXTURE_EXTRACTION_OPTIONS.items():
        print(f"{option_number}. {config['name']} - {config['description']}")


def feature_option_uses_texture(option: str) -> bool:
    if option not in FEATURE_SPACE_OPTIONS:
        return False
    _, parts = FEATURE_SPACE_OPTIONS[option]
    return "texture" in parts


def ask_texture_extraction_method(default: str = "1") -> str:
    global selected_texture_method_saved

    print_texture_extraction_menu()
    text = input(f"Enter the texture option number [{default}]: ").strip()
    if not text:
        text = default
    if text not in TEXTURE_EXTRACTION_OPTIONS:
        print("Invalid texture option selected. Using option 1: Gabor filter bank.")
        text = "1"

    selected_texture_method_saved = str(TEXTURE_EXTRACTION_OPTIONS[text]["key"])
    print(f"Using texture extraction method: {texture_method_display_name(selected_texture_method_saved)}")

    return selected_texture_method_saved


def ask_mean_shift_processing_mode(default: str = "1") -> str:
    print_mean_shift_processing_menu()
    text = input(f"Enter the option number [{default}]: ").strip()
    if not text:
        return default
    if text not in MEAN_SHIFT_PROCESSING_OPTIONS:
        print("Invalid Mean Shift processing option selected. Using option 1.")
        return "1"
    return text


def select_feature_space(
    extractor: FeatureExtractor,
    option: str,
) -> tuple[np.ndarray | None, np.ndarray | None, tuple[str, ...] | None]:
    global selectedoptionSaved

    if option not in FEATURE_SPACE_OPTIONS:
        return None, None, None

    selectedoptionSaved, parts = FEATURE_SPACE_OPTIONS[option]
    print(f"Using feature combination: {selectedoptionSaved.name}")
    if "texture" in parts:
        print(f"Texture method: {texture_method_display_name(extractor.texture_method)}")
    features_hwf, X = extractor.create_feature_space(parts)
    return features_hwf, X, parts


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


def run_mean_shift_with_processing_mode(
    image: np.ndarray,
    X: np.ndarray,
    parts: Iterable[str],
    bandwidths: Sequence[float],
    processing_mode: str,
) -> None:
    processed_image, processed_X, algorithm_name, use_sampling = prepare_mean_shift_inputs(
        image=image,
        X=X,
        parts=parts,
        processing_mode=processing_mode,
    )
    mean_shift_segmentation(
        processed_image,
        processed_X,
        bandwidths=bandwidths,
        algorithm_name=algorithm_name,
        use_sampling=use_sampling,
    )


def run_clustering_for_current_feature_space(
    image: np.ndarray,
    X: np.ndarray,
    parts: Iterable[str],
    clustering_option: str,
) -> None:
    if clustering_option == "1":
        apply_k_means_segmentation(image, X, k=ask_k())
    elif clustering_option == "2":
        processing_mode = ask_mean_shift_processing_mode()
        run_mean_shift_with_processing_mode(
            image,
            X,
            parts,
            bandwidths=ask_bandwidths(),
            processing_mode=processing_mode,
        )
    elif clustering_option == "3":
        apply_k_means_segmentation(image, X, k=ask_k())
        processing_mode = ask_mean_shift_processing_mode()
        run_mean_shift_with_processing_mode(
            image,
            X,
            parts,
            bandwidths=ask_bandwidths(),
            processing_mode=processing_mode,
        )
    else:
        raise ValueError("Invalid clustering option selected.")


def do_all(
    image: np.ndarray,
    extractor: FeatureExtractor,
    mean_shift_processing_mode: str = "1",
) -> None:
    """Run all 15 feature-space combinations with cached feature extraction."""
    global selectedoptionSaved

    for option_number, (selected_option, parts) in FEATURE_SPACE_OPTIONS.items():
        selectedoptionSaved = selected_option
        print("=" * 80)
        print(f"Running combination {option_number}: {selectedoptionSaved.name}")
        if "texture" in parts:
            print(f"Texture method: {texture_method_display_name(extractor.texture_method)}")

        _, X = extractor.create_feature_space(parts)
        apply_k_means_segmentation(image, X, k=5)
        run_mean_shift_with_processing_mode(
            image,
            X,
            parts,
            bandwidths=(0.8, 1.5),
            processing_mode=mean_shift_processing_mode,
        )


def main() -> None:
    image = load_image()
    if image is None:
        return

    print_feature_space_menu()
    option = input("Enter the option number: ").strip()

    if option == "0" or option not in FEATURE_SPACE_OPTIONS or feature_option_uses_texture(option):
        ask_texture_extraction_method()

    extractor = FeatureExtractor(
        image,
        texture_method=selected_texture_method_saved,
    )

    if option == "0":
        mean_shift_processing_mode = ask_mean_shift_processing_mode()
        do_all(image, extractor, mean_shift_processing_mode=mean_shift_processing_mode)
        return

    _, X, parts = select_feature_space(extractor, option)
    if X is None or parts is None:
        print("Invalid option selected. Running all combinations instead.")
        mean_shift_processing_mode = ask_mean_shift_processing_mode()
        do_all(image, extractor, mean_shift_processing_mode=mean_shift_processing_mode)
        return

    print("Select a clustering algorithm:")
    print("1. KMeans (library MiniBatchKMeans)")
    print("2. Mean Shift (library sklearn MeanShift)")
    print("3. Both KMeans and Mean Shift")

    clustering_option = input("Enter the option number: ").strip()
    run_clustering_for_current_feature_space(image, X, parts, clustering_option)

    if SHOW_WINDOWS:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
