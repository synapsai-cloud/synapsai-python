from PIL import Image
import base64
import os
from typing import Any
import numpy as np
import _io

def process_image_input(image: Any):
    """Process image input (file path, bytes, base64, URL, or lists of those)"""

    if image is None:
        return None

    # Handle batched inputs
    if isinstance(image, list):
        return [process_image_input(img) for img in image]

    # Single input handling
    if isinstance(image, str):
        # URL
        if image.startswith(("http://", "https://")):
            return image

        # File path
        if os.path.isfile(image):
            try:
                with open(image, "rb") as f:
                    image_bytes = f.read()
            except FileNotFoundError:
                raise ValueError("File not found")
            return base64.b64encode(image_bytes).decode("utf-8")

        # Assume base64
        return image

    if isinstance(image, bytes):
        return base64.b64encode(image).decode("utf-8")

    if isinstance(image, Image.Image):
        return base64.b64encode(image.tobytes()).decode("utf-8")

    raise ValueError(
        "Image must be a file path, bytes, PIL.Image, URL, base64 string, or a list of these"
    )

def process_audio_input(file):
    """Process audio file input"""
    if file is None:
        return None

    # Handle batched inputs
    if isinstance(file, list):
        return [process_audio_input(f) for f in file]

    if isinstance(file, str):
        # Check if it's a file path
        if os.path.isfile(file):
            try:
                with open(file, "rb") as f:
                    audio_bytes = f.read()
            except FileNotFoundError:
                raise ValueError("File not found")
            return base64.b64encode(audio_bytes).decode("utf-8")
        # Assume it's already base64
        return file
    elif isinstance(file, bytes):
        return base64.b64encode(file).decode("utf-8")
    elif isinstance(file, np.ndarray):
        return base64.b64encode(file.tobytes()).decode("utf-8")
    elif isinstance(file, _io.BufferedReader):
        return base64.b64encode(file.read()).decode("utf-8")
    else:
        raise ValueError("File must be a file path, bytes, or base64 string")