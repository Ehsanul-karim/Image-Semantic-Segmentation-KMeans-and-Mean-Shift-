#first ask user about which image to load

#e.g. Img01, Img02, etc.

#Load that image from outputs/{imgName}/imgName.jpg or imgName.png

# extract some features from the image, for example: color space (e.g., RGB, LAB, HSV), spatial position, texture features (e.g., Local Binary Patterns, Gabor Filter Bank, Leung-Malik filter Bank, etc.), etc.
# visualize the texture filters and their convolution results on the image using cv2.imshow and cv2.waitKey(0)


import os
import cv2
import numpy as np
    
def load_image():
    img_name = input("Enter the image name (e.g., Img01, Img02): ")
    img_path_jpg = os.path.join("outputs", img_name, f"{img_name}.jpg")
    img_path_png = os.path.join("outputs", img_name, f"{img_name}.png")

    if os.path.exists(img_path_jpg):
        img = cv2.imread(img_path_jpg)
        print(f"Loaded image from {img_path_jpg}")
        return img
    elif os.path.exists(img_path_png):
        img = cv2.imread(img_path_png)
        print(f"Loaded image from {img_path_png}")
        return img
    else:
        print("Image not found. Please check the name and try again.")
        return None
    
def generate_gabor_kernels(ksize,wevelengths, orientations, phase_offsets, std_dev, spatial_aspect_ratio):
    kernels = []
    for theta in orientations:
        for phi in phase_offsets:
            for wavelength in wevelengths:
                kernel = cv2.getGaborKernel(
                    (ksize, ksize),
                    std_dev,
                    theta,
                    wavelength,
                    spatial_aspect_ratio,
                    phi,
                )
                kernels.append(kernel)
    return kernels
    
def visualize_gabor_texture_kernels(kernels):
    for idx, kernel in enumerate(kernels):
        cv2.imshow(f"Kernel {idx}", kernel)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def convolution(image, kernel):
    image_height = image.shape[0]
    image_width = image.shape[1]

    kernel_height = kernel.shape[0]
    kernel_width = kernel.shape[1]

    centerx = kernel_height // 2
    centery = kernel_width // 2

    output_image = np.zeros((image_height, image_width), dtype=np.float64)

    for i in range(image_height):
        for j in range(image_width):
            total = 0

            for k in range(kernel_height):
                for l in range(kernel_width):
                    image_x = i - centerx + k
                    image_y = j - centery + l

                    if 0 <= image_x < image_height and 0 <= image_y < image_width:
                        total += (
                            kernel[kernel_height - k - 1][kernel_width - l - 1]
                            * image[image_x][image_y]
                        )

            output_image[i][j] = total

    return output_image

def show_feature(name, feature):
    display = cv2.normalize(feature, None, 0, 255, cv2.NORM_MINMAX)
    display = display.astype(np.uint8)
    #cv2.imshow(name, display)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

def visualize_gabor_texture_filters(image, filters):
    feature_space = []
    for idx, kernel in enumerate(filters):
        # feature_image = convolution(image, kernel)
        feature_image = cv2.filter2D(image.astype(np.float32), cv2.CV_32F, kernel)
        feature_space.append(feature_image)
        show_feature(f"Gabor Filter {idx}", feature_image)
    return feature_space


def create_rgb_hsv_lab_xy_texture_feature_space(image):
    # Example: Convert the image to different color spaces
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Example: Generate Gabor kernels
    ksize = 31
    wavelengths = [4, 8, 16, 32]
    orientations = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    phase_offsets = [0, 0.8]
    std_dev = 4.0
    spatial_aspect_ratio = 0.1

    gabor_kernels = generate_gabor_kernels(ksize, wavelengths, orientations, phase_offsets, std_dev, spatial_aspect_ratio)

    # Visualize the Gabor kernels
    visualize_gabor_texture_kernels(gabor_kernels)

    # Visualize the convolution results of the Gabor filters on the image
    feature_space = visualize_gabor_texture_filters(gray_image, gabor_kernels)
    #Add R channel, G channel, B channel, L channel, A channel, H channel, S channel, V channel to the feature space
    feature_space.extend([
        rgb_image[:, :, 0],  # R
        rgb_image[:, :, 1],  # G
        rgb_image[:, :, 2],  # B
        lab_image[:, :, 0],  # L
        lab_image[:, :, 1],  # A
        lab_image[:, :, 2],  # B
        hsv_image[:, :, 0],  # H
        hsv_image[:, :, 1],  # S
        hsv_image[:, :, 2],  # V
    ])

    h, w = gray_image.shape
    x_coords, y_coords = np.meshgrid(
        np.arange(w, dtype=np.float32),
        np.arange(h, dtype=np.float32)
    )

    x_coords /= max(w - 1, 1)
    y_coords /= max(h - 1, 1)
    feature_space.append(x_coords)
    feature_space.append(y_coords)

    features_hwf = np.stack(feature_space, axis=-1).astype(np.float32)

    X = features_hwf.reshape(-1, features_hwf.shape[-1])
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-6)

    return features_hwf, X

def create_rgb_only_feature_space(image):
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    feature_space = [
        rgb_image[:, :, 0],  # R
        rgb_image[:, :, 1],  # G
        rgb_image[:, :, 2],  # B
    ]
    features_hwf = np.stack(feature_space, axis=-1).astype(np.float32)
    X = features_hwf.reshape(-1, features_hwf.shape[-1])
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-6)
    return features_hwf, X

def create_hsv_only_feature_space(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    feature_space = [
        hsv_image[:, :, 0],  # H
        hsv_image[:, :, 1],  # S
        hsv_image[:, :, 2],  # V
    ]
    features_hwf = np.stack(feature_space, axis=-1).astype(np.float32)
    X = features_hwf.reshape(-1, features_hwf.shape[-1])
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-6)
    return features_hwf, X

def create_lab_only_feature_space(image):
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    feature_space = [
        lab_image[:, :, 0],  # L
        lab_image[:, :, 1],  # A
        lab_image[:, :, 2],  # B
    ]
    features_hwf = np.stack(feature_space, axis=-1).astype(np.float32)
    X = features_hwf.reshape(-1, features_hwf.shape[-1])
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-6)
    return features_hwf, X

def create_xy_only_feature_space(image):
    h, w = image.shape[:2]
    x_coords, y_coords = np.meshgrid(
        np.arange(w, dtype=np.float32),
        np.arange(h, dtype=np.float32)
    )
    x_coords /= max(w - 1, 1)
    y_coords /= max(h - 1, 1)
    feature_space = [x_coords, y_coords]
    features_hwf = np.stack(feature_space, axis=-1).astype(np.float32)
    X = features_hwf.reshape(-1, features_hwf.shape[-1])
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-6)
    return features_hwf, X

def create_texture_only_feature_space(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ksize = 31
    wavelengths = [4, 8, 16, 32]
    orientations = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    phase_offsets = [0, 0.8]
    std_dev = 4.0
    spatial_aspect_ratio = 0.1

    gabor_kernels = generate_gabor_kernels(ksize, wavelengths, orientations, phase_offsets, std_dev, spatial_aspect_ratio)
    visualize_gabor_texture_kernels(gabor_kernels)
    feature_space = visualize_gabor_texture_filters(gray_image, gabor_kernels)

    features_hwf = np.stack(feature_space, axis=-1).astype(np.float32)
    X = features_hwf.reshape(-1, features_hwf.shape[-1])
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-6)
    return features_hwf, X

def create_rgb_xy_feature_space(image):
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w = image.shape[:2]
    x_coords, y_coords = np.meshgrid(
        np.arange(w, dtype=np.float32),
        np.arange(h, dtype=np.float32)
    )
    x_coords /= max(w - 1, 1)
    y_coords /= max(h - 1, 1)

    feature_space = [
        rgb_image[:, :, 0],  # R
        rgb_image[:, :, 1],  # G
        rgb_image[:, :, 2],  # B
        x_coords,
        y_coords,
    ]
    features_hwf = np.stack(feature_space, axis=-1).astype(np.float32)
    X = features_hwf.reshape(-1, features_hwf.shape[-1])
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-6)
    return features_hwf, X

def create_rgb_texture_feature_space(image):
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    ksize = 31
    wavelengths = [4, 8, 16, 32]
    orientations = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    phase_offsets = [0, 0.8]
    std_dev = 4.0
    spatial_aspect_ratio = 0.1

    gabor_kernels = generate_gabor_kernels(ksize, wavelengths, orientations, phase_offsets, std_dev, spatial_aspect_ratio)
    visualize_gabor_texture_kernels(gabor_kernels)
    texture_features = visualize_gabor_texture_filters(gray_image, gabor_kernels)

    feature_space = [
        rgb_image[:, :, 0],  # R
        rgb_image[:, :, 1],  # G
        rgb_image[:, :, 2],  # B
    ] + texture_features

    features_hwf = np.stack(feature_space, axis=-1).astype(np.float32)
    X = features_hwf.reshape(-1, features_hwf.shape[-1])
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-6)
    return features_hwf, X

def create_hsv_xy_feature_space(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, w = image.shape[:2]
    x_coords, y_coords = np.meshgrid(
        np.arange(w, dtype=np.float32),
        np.arange(h, dtype=np.float32)
    )
    x_coords /= max(w - 1, 1)
    y_coords /= max(h - 1, 1)

    feature_space = [
        hsv_image[:, :, 0],  # H
        hsv_image[:, :, 1],  # S
        hsv_image[:, :, 2],  # V
        x_coords,
        y_coords,
    ]
    features_hwf = np.stack(feature_space, axis=-1).astype(np.float32)
    X = features_hwf.reshape(-1, features_hwf.shape[-1])
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-6)
    return features_hwf, X

def create_lab_xy_feature_space(image):
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    h, w = image.shape[:2]
    x_coords, y_coords = np.meshgrid(
        np.arange(w, dtype=np.float32),
        np.arange(h, dtype=np.float32)
    )
    x_coords /= max(w - 1, 1)
    y_coords /= max(h - 1, 1)

    feature_space = [
        lab_image[:, :, 0],  # L
        lab_image[:, :, 1],  # A
        lab_image[:, :, 2],  # B
        x_coords,
        y_coords,
    ]
    features_hwf = np.stack(feature_space, axis=-1).astype(np.float32)
    X = features_hwf.reshape(-1, features_hwf.shape[-1])
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-6)
    return features_hwf, X

def create_hsv_texture_feature_space(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    ksize = 31
    wavelengths = [4, 8, 16, 32]
    orientations = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    phase_offsets = [0, 0.8]
    std_dev = 4.0
    spatial_aspect_ratio = 0.1

    gabor_kernels = generate_gabor_kernels(ksize, wavelengths, orientations, phase_offsets, std_dev, spatial_aspect_ratio)
    visualize_gabor_texture_kernels(gabor_kernels)
    texture_features = visualize_gabor_texture_filters(gray_image, gabor_kernels)

    feature_space = [
        hsv_image[:, :, 0],  # H
        hsv_image[:, :, 1],  # S
        hsv_image[:, :, 2],  # V
    ] + texture_features

    features_hwf = np.stack(feature_space, axis=-1).astype(np.float32)
    X = features_hwf.reshape(-1, features_hwf.shape[-1])
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-6)
    return features_hwf, X

def create_lab_texture_feature_space(image):
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    ksize = 31
    wavelengths = [4, 8, 16, 32]
    orientations = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    phase_offsets = [0, 0.8]
    std_dev = 4.0
    spatial_aspect_ratio = 0.1

    gabor_kernels = generate_gabor_kernels(ksize, wavelengths, orientations, phase_offsets, std_dev, spatial_aspect_ratio)
    visualize_gabor_texture_kernels(gabor_kernels)
    texture_features = visualize_gabor_texture_filters(gray_image, gabor_kernels)

    feature_space = [
        lab_image[:, :, 0],  # L
        lab_image[:, :, 1],  # A
        lab_image[:, :, 2],  # B
    ] + texture_features

    features_hwf = np.stack(feature_space, axis=-1).astype(np.float32)
    X = features_hwf.reshape(-1, features_hwf.shape[-1])
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-6)
    return features_hwf, X

#Randomly select K data points (pixels) to act as the initial cluster centers
def initialize_k_centroids(X, k):
    np.random.seed(42)  # For reproducibility
    random_indices = np.random.choice(X.shape[0], size=k, replace=False)
    centroids = X[random_indices]
    return centroids

#For every single pixel in the flattened array, calculate the Euclidean distance to each of the K centroids
def calculate_distances(X, centroids):
    distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
    return distances

#Assign each pixel to the nearest cluster centroid (the one with the shortest distance
def assign_clusters(X, centroids):
    distances = calculate_distances(X, centroids)
    cluster_labels = np.argmin(distances, axis=1)
    return cluster_labels

#For each of the K clusters, compute the average value (mean) of all the pixels assigned to it.
# Update the position of the K centroids to match these new mean values.
def update_centroids(X, cluster_labels, k):
    new_centroids = np.zeros((k, X.shape[1]), dtype=np.float32)
    for i in range(k):
        cluster_points = X[cluster_labels == i]
        if len(cluster_points) > 0:
            new_centroids[i] = cluster_points.mean(axis=0)
    return new_centroids


def postprocess_k_Means_segmented_image(label_image,k):
    
    # 1. Smooth small noisy label changes
    label_image = cv2.medianBlur(label_image.astype(np.uint8), 5)

    # 2. Remove very small connected components
    cleaned = label_image.copy()

    for label in range(k):
        mask = (label_image == label).astype(np.uint8)

        num_components, components, stats, _ = cv2.connectedComponentsWithStats(
            mask,
            connectivity=8
        )

        for component_id in range(1, num_components):
            area = stats[component_id, cv2.CC_STAT_AREA]

            if area < 100:  # Threshold for small components i.e min_area = 100. minimum allowed connected componenet area = 100
                component_mask = components == component_id

                # Assign small component to nearest surrounding label using dilation
                dilated = cv2.dilate(
                    component_mask.astype(np.uint8),
                    np.ones((3, 3), np.uint8),
                    iterations=1
                )

                border_mask = (dilated == 1) & (~component_mask)

                surrounding_labels = label_image[border_mask]

                if len(surrounding_labels) > 0:
                    new_label = np.bincount(surrounding_labels).argmax()
                    cleaned[component_mask] = new_label

    return cleaned

def apply_k_means_segmentation(image, X):
    # As the goal is segmenation, use this feature space and apply clustering algorithms kmeans. Here the image will be displayed to user and then a waitkey(0) and then he will be asked to enter the approximate value of k. Then using Kmeans to perform clustring
    cv2.imshow("Loaded Image", image)
    cv2.waitKey(0)
    print("Enter the number of segments (k) for this image:")
    k = int(input())
    cv2.destroyAllWindows()

    # Perform KMeans clustering, i can't use sklearn so need to built my own
    centroids = initialize_k_centroids(X, k)
    #Repeat the Assignment and Update steps iteratively until the centroids stop moving or a specific maximum number of iterations is reached
    for iteration in range(100):  
        cluster_labels = assign_clusters(X, centroids)
        new_centroids = update_centroids(X, cluster_labels, k)

        if np.allclose(centroids, new_centroids):
            print(f"KMeans converged after {iteration} iterations.")
            break

        centroids = new_centroids

    #Replace the color of each pixel in the original image with the color of its final assigned cluster centroid.
    label_image = cluster_labels.reshape(image.shape[:2])
    label_display = cv2.normalize(label_image.astype(np.uint8), None, 0, 255, cv2.NORM_MINMAX)
    label_display = label_display.astype(np.uint8)
    label_display = cv2.applyColorMap(label_display, cv2.COLORMAP_JET)

    cv2.imshow("Segmented Image (KMeans)", label_display)        
    cv2.waitKey(0)
    
    #apply some postprocessing to the segmented image.
    postprocessed_labels = postprocess_k_Means_segmented_image(label_image,k)
    postprocessed_display = cv2.normalize(postprocessed_labels.astype(np.uint8), None, 0, 255, cv2.NORM_MINMAX)
    postprocessed_display = postprocessed_display.astype(np.uint8)
    postprocessed_display = cv2.applyColorMap(postprocessed_display, cv2.COLORMAP_JET)
    cv2.imshow("Postprocessed Segmented Image (KMeans)", postprocessed_display)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

import math


def gaussian_kernel_weights(dist2, bandwidth):
    h2 = bandwidth * bandwidth
    weights = np.exp(-dist2 / (2.0 * h2))
    weights[dist2 > (3 * bandwidth) ** 2] = 0.0
    return weights


def epanechnikov_kernel_weights(dist2, bandwidth):
    h2 = bandwidth * bandwidth
    weights = np.maximum(0.0, 1.0 - (dist2 / h2))
    weights[dist2 > h2] = 0.0
    return weights


def compute_kernel_weights(dist2, bandwidth, kernel_type):
    if kernel_type == "gaussian":
        return gaussian_kernel_weights(dist2, bandwidth)

    else:
        return epanechnikov_kernel_weights(dist2, bandwidth)


def bandwidth_diagnostics(X, sample_size=2000, kth=30, seed=42):
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    sample_size = min(sample_size, n)

    idx = rng.choice(n, size=sample_size, replace=False)
    S = X[idx].astype(np.float32)

    diff = S[:, None, :] - S[None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))

    np.fill_diagonal(dist, np.inf)

    kth_dist = np.partition(dist, kth, axis=1)[:, kth]

    print("kNN distance percentiles:")
    for p in [10, 25, 50, 75, 90]:
        print(f"{p}%: {np.percentile(kth_dist, p):.4f}")

    print("\nSuggested bandwidth candidates:")
    print(np.percentile(kth_dist, [25, 50, 75, 90]))

    return kth_dist


def mode_distance_diagnostics(modes, sample_size=3000, seed=42):
    rng = np.random.default_rng(seed)
    n = modes.shape[0]
    sample_size = min(sample_size, n)

    idx = rng.choice(n, size=sample_size, replace=False)
    S = modes[idx].astype(np.float32)

    diff = S[:, None, :] - S[None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dist, np.inf)

    nearest = np.min(dist, axis=1)

    print("Nearest-mode distance percentiles:")
    for p in [10, 25, 50, 75, 90, 95]:
        print(f"{p}%: {np.percentile(nearest, p):.4f}")

    return nearest

def shift_single_point(point, X, bandwidth, kernel_type, max_iter=30, stop_thresh=1e-3):

    current_point = point.astype(np.float32).copy()

    for _ in range(max_iter):
        diff = X - current_point
        dist2 = np.sum(diff * diff, axis=1)

        weights = compute_kernel_weights(dist2, bandwidth, kernel_type)
        weight_sum = np.sum(weights)

        if weight_sum <= 1e-12:
            break

        new_point = np.sum(X * weights[:, np.newaxis], axis=0) / weight_sum

        shift_distance = np.linalg.norm(new_point - current_point)

        current_point = new_point.astype(np.float32)

        if shift_distance < stop_thresh:
            break

    return current_point


def merge_modes(modes, bandwidth, merge_threshold):

    cluster_centers = []
    cluster_counts = []
    labels = np.full(modes.shape[0], -1, dtype=np.int32)

    for i, mode in enumerate(modes):
        if len(cluster_centers) == 0:
            cluster_centers.append(mode.copy())
            cluster_counts.append(1)
            labels[i] = 0
            continue

        centers_array = np.array(cluster_centers, dtype=np.float32)
        distances = np.linalg.norm(centers_array - mode, axis=1)

        nearest_cluster = np.argmin(distances)

        if distances[nearest_cluster] < merge_threshold:
            labels[i] = nearest_cluster

            count = cluster_counts[nearest_cluster]
            cluster_centers[nearest_cluster] = (
                cluster_centers[nearest_cluster] * count + mode
            ) / (count + 1)
            cluster_counts[nearest_cluster] += 1

        else:
            new_cluster_id = len(cluster_centers)
            cluster_centers.append(mode.copy())
            cluster_counts.append(1)
            labels[i] = new_cluster_id

    return labels, np.array(cluster_centers, dtype=np.float32)


def labels_to_segmented_image(image, labels):

    h, w = image.shape[:2]
    label_image = labels.reshape(h, w)

    segmented_image = np.zeros_like(image, dtype=np.float32)

    unique_labels = np.unique(labels)

    for label in unique_labels:
        mask = label_image == label

        if np.any(mask):
            mean_color = image[mask].mean(axis=0)
            segmented_image[mask] = mean_color

    return label_image, segmented_image.astype(np.uint8)


def postprocess_labels(label_image):

    cleaned = label_image.copy()
    unique_labels = np.unique(label_image)

    for label in unique_labels:
        mask = (label_image == label).astype(np.uint8)

        num_components, components, stats, _ = cv2.connectedComponentsWithStats(
            mask,
            connectivity=8
        )

        for component_id in range(1, num_components):
            area = stats[component_id, cv2.CC_STAT_AREA]

            if area < 100:
                component_mask = components == component_id

                dilated = cv2.dilate(
                    component_mask.astype(np.uint8),
                    np.ones((3, 3), np.uint8),
                    iterations=1
                )

                border_mask = (dilated == 1) & (~component_mask)

                surrounding_labels = label_image[border_mask]

                if surrounding_labels.size > 0:
                    new_label = np.bincount(surrounding_labels).argmax()
                    cleaned[component_mask] = new_label

    return cleaned

def mean_shift_segmentation(image, X, bandwidths, max_iter=30):
    
    h, w = image.shape[:2]
    n_pixels = h * w

    if X.shape[0] != n_pixels:
        print(f"X has {X.shape[0]} rows, but image has {n_pixels} pixels. "
            "X must have one row per pixel.")
        exit()

    X = X.astype(np.float32)
    kernel_types = ["gaussian", "epanechnikov"]

    for bandwidth in bandwidths:
        for kernel_type in kernel_types:
            stop_thresh =  0.01 * bandwidth
            modes = np.zeros_like(X, dtype=np.float32)

            for i in range(X.shape[0]):
                modes[i] = shift_single_point(
                    point=X[i],
                    X=X,
                    bandwidth=bandwidth,
                    kernel_type=kernel_type,
                    max_iter=max_iter,
                    stop_thresh=stop_thresh
                )
            nearest_mode_distance = mode_distance_diagnostics(modes)
            print("Enter merge threshold based on the above nearest-mode distances:")
            merge_threshold = float(input("Merge threshold: "))
            input("Press Anything")
            labels, cluster_centers = merge_modes(
                modes,
                bandwidth=bandwidth,
                merge_threshold=merge_threshold
            )

            label_image, segmented_image = labels_to_segmented_image(image, labels)
            cv2.imshow(
                f"Mean Shift - {kernel_type} - h={bandwidth}",
                segmented_image
            )
            cv2.waitKey(0)

            label_image = postprocess_labels(label_image)
            labels = label_image.reshape(-1)
            label_image, segmented_image = labels_to_segmented_image(image, labels)

            cv2.imshow(
                f"Mean Shift Post Processed - {kernel_type} - h={bandwidth}",
                segmented_image
            )
            cv2.waitKey(0)
            cv2.destroyAllWindows()

if __name__ == "__main__":
    image = load_image()
    if image is not None:
        #display a list of available option and ask user to select one of the options for feature space creation
        
        print("Select a feature space creation method:")
        print("1. RGB + HSV + LAB + XY + Texture")
        print("2. RGB only")
        print("3. HSV only")
        print("4. LAB only")
        print("5. XY only")
        print("6. Texture only")
        print("7. RGB + XY")
        print("8. RGB + Texture")
        print("9. HSV + XY")
        print("10. LAB + XY")
        print("11. HSV + Texture")
        print("12. LAB + Texture")
        option = input("Enter the option number: ")
        if option == "1":
            feature_space, X = create_rgb_hsv_lab_xy_texture_feature_space(image)
        elif option == "2":
            feature_space, X = create_rgb_only_feature_space(image)
        elif option == "3":
            feature_space, X = create_hsv_only_feature_space(image)
        elif option == "4":
            feature_space, X = create_lab_only_feature_space(image)
        elif option == "5":
            feature_space, X = create_xy_only_feature_space(image)
        elif option == "6":
            feature_space, X = create_texture_only_feature_space(image)
        elif option == "7":
            feature_space, X = create_rgb_xy_feature_space(image)
        elif option == "8":
            feature_space, X = create_rgb_texture_feature_space(image)
        elif option == "9":
            feature_space, X = create_hsv_xy_feature_space(image)
        elif option == "10":
            feature_space, X = create_lab_xy_feature_space(image)
        elif option == "11":
            feature_space, X = create_hsv_texture_feature_space(image)
        elif option == "12":
            feature_space, X = create_lab_texture_feature_space(image)
        else:
            print("Invalid option selected. Please try again.")
            exit()

        print("Select a clustering algorithm:")
        print("1. KMeans")
        print("2. Mean Shift")

        clustering_option = input("Enter the option number: ")

        if clustering_option == "1":
            apply_k_means_segmentation(image, X)
        elif clustering_option == "2":
            kth_dist = bandwidth_diagnostics(X, sample_size=2000, kth=30)
            print("Enter two bandwidth values based on the above suggestion:")
            b1 = float(input("Bandwidth small: "))
            b2 = float(input("Bandwidth large: "))
            bandwidths = [b1, b2]
            mean_shift_segmentation(image, X, bandwidths)
            pass
        else:
            print("Invalid option selected. Please try again.")
            exit()

        #still need to implement mean shift algorithm and finding the appropiate bandwidth parameter for each feature space. at least 2 different value is required. Use Gaussian as well as Epanechnikov kernel
        