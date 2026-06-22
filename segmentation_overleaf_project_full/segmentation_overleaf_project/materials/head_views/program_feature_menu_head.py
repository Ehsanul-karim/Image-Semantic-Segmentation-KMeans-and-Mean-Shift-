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

...
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