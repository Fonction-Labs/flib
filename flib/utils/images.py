import os
import cv2
import io
import base64
from PIL import Image


def write_file_to_temp_folder(file: str, temp_dir: str) -> str:
    """
    Writes a Streamlit File object to the session temporary folder, and returns the filepath.
    """
    path = os.path.join(temp_dir, file.name)
    with open(path, "wb") as f:
        f.write(file.read())
    return path


def load_image(image_path: str, use_cv2: bool = False) -> (cv2.typing.MatLike | Image.Image):
    if use_cv2:
        return cv2.imread(image_path)
    return Image.open(image_path)


def encode_image_base64(image: Image.Image) -> str:
    """
    Takes a PIL image, converts it to bytes, and encodes it with base 64.
    """
    img_byte_array = io.BytesIO()
    image.save(img_byte_array, format="png")  # image.format)
    img_bytes = img_byte_array.getvalue()
    return base64.b64encode(img_bytes).decode("utf-8")
