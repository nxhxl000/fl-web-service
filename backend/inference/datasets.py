"""Class names + image preprocessing per dataset.

Hardcoded copy of metadata that lives in fl_app — duplicating it here avoids
pulling matplotlib/flwr into the API process just for inference.
"""

from __future__ import annotations

from typing import Callable

from PIL import Image
from torchvision import transforms


CIFAR100_FINE_NAMES: list[str] = [
    "apple", "aquarium_fish", "baby", "bear", "beaver", "bed", "bee", "beetle", "bicycle", "bottle",
    "bowl", "boy", "bridge", "bus", "butterfly", "camel", "can", "castle", "caterpillar", "cattle",
    "chair", "chimpanzee", "clock", "cloud", "cockroach", "couch", "crab", "crocodile", "cup", "dinosaur",
    "dolphin", "elephant", "flatfish", "forest", "fox", "girl", "hamster", "house", "kangaroo", "keyboard",
    "lamp", "lawn_mower", "leopard", "lion", "lizard", "lobster", "man", "maple_tree", "motorcycle", "mountain",
    "mouse", "mushroom", "oak_tree", "orange", "orchid", "otter", "palm_tree", "pear", "pickup_truck", "pine_tree",
    "plain", "plate", "poppy", "porcupine", "possum", "rabbit", "raccoon", "ray", "road", "rocket",
    "rose", "sea", "seal", "shark", "shrew", "skunk", "skyscraper", "snail", "snake", "spider",
    "squirrel", "streetcar", "sunflower", "sweet_pepper", "table", "tank", "telephone", "television", "tiger", "tractor",
    "train", "trout", "tulip", "turtle", "wardrobe", "whale", "willow_tree", "wolf", "woman", "worm",
]

# 38 classes; canonical PlantVillage label order (matches HuggingFace dataset).
PLANTVILLAGE_NAMES: list[str] = [
    "Apple___Apple_scab", "Apple___Black_rot", "Apple___Cedar_apple_rust", "Apple___healthy",
    "Blueberry___healthy", "Cherry___Powdery_mildew", "Cherry___healthy",
    "Corn___Cercospora_leaf_spot Gray_leaf_spot", "Corn___Common_rust", "Corn___Northern_Leaf_Blight",
    "Corn___healthy", "Grape___Black_rot", "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)", "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)", "Peach___Bacterial_spot", "Peach___healthy",
    "Pepper,_bell___Bacterial_spot", "Pepper,_bell___healthy", "Potato___Early_blight",
    "Potato___Late_blight", "Potato___healthy", "Raspberry___healthy", "Soybean___healthy",
    "Squash___Powdery_mildew", "Strawberry___Leaf_scorch", "Strawberry___healthy",
    "Tomato___Bacterial_spot", "Tomato___Early_blight", "Tomato___Late_blight", "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot", "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot", "Tomato___Tomato_Yellow_Leaf_Curl_Virus", "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy",
]


DATASET_INFO: dict[str, dict] = {
    "cifar100": {
        "mean": (0.5071, 0.4866, 0.4409),
        "std": (0.2673, 0.2564, 0.2762),
        "size": 32,
        "class_names": CIFAR100_FINE_NAMES,
    },
    "plantvillage": {
        "mean": (0.485, 0.456, 0.406),
        "std": (0.229, 0.224, 0.225),
        "size": 224,
        "class_names": PLANTVILLAGE_NAMES,
    },
}


def get_class_names(dataset: str) -> list[str]:
    info = DATASET_INFO.get(dataset)
    if info is None:
        raise ValueError(f"Unknown dataset {dataset!r}")
    return info["class_names"]


def get_eval_transform(dataset: str) -> Callable:
    """Mirrors fl_app/data.py:_eval_transform — eval-time preprocessing."""
    info = DATASET_INFO.get(dataset)
    if info is None:
        raise ValueError(f"Unknown dataset {dataset!r}")
    if dataset == "cifar100":
        return transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(info["mean"], info["std"]),
        ])
    return transforms.Compose([
        transforms.Resize((info["size"], info["size"])),
        transforms.ToTensor(),
        transforms.Normalize(info["mean"], info["std"]),
    ])


def open_rgb(image_bytes: bytes) -> Image.Image:
    """Open an arbitrary image as RGB (handles transparency, palettes, etc)."""
    import io
    img = Image.open(io.BytesIO(image_bytes))
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img
