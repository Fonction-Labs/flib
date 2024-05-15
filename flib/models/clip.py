import torch
import clip
from PIL import Image

from .base import BaseModel

class LocalCLIPModel(BaseModel):
    # https://github.com/openai/CLIP

    def __init__(self):
        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else ("mps" if torch.backends.mps.is_available() else "cpu")
        )
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        self.embedding_vector_size = 512

    def run(self, image: Image.Image) -> list[float]:
        image = self.preprocess(image).unsqueeze(0).to(self.device)
        # text = clip.tokenize(["a diagram", "a dog", "a cat"]).to(device)
        with torch.no_grad():
            image_features = self.model.encode_image(image)
            # text_features = self.model.encode_text(text)
        image_features = list(image_features.detach().cpu().numpy()[0])
        return image_features
