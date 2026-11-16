import joblib
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from be.config import settings


class MLService:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.initialized = False
        return cls._instance

    def initialize(self):
        if self.initialized:
            return

        print("[INFO] Ładowanie modelu CLIP...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CLIPModel.from_pretrained(
            settings.CLIP_MODEL_ID,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
        ).to(self.device)
        self.model.eval()
        self.processor = CLIPProcessor.from_pretrained(settings.CLIP_MODEL_ID)

        print("[INFO] Wczytywanie zserializowanego UMAP...")
        if not settings.UMAP_MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Brak modelu UMAP: {settings.UMAP_MODEL_PATH}. Wykonaj Etap 2."
            )
        self.umap_reducer = joblib.load(settings.UMAP_MODEL_PATH)

        self.initialized = True
        print(f"[SUKCES] ML Service gotowy. Backend inferencji: {self.device.upper()}")

    def infer_image(self, image: Image.Image) -> tuple[float, float, float]:
        inputs = self.processor(images=[image.convert("RGB")], return_tensors="pt")
        gpu_inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with (
            torch.amp.autocast(self.device)
            if self.device == "cuda"
            else torch.no_grad(),
            torch.no_grad(),
        ):
            outputs = self.model.get_image_features(**gpu_inputs)
            img_feats = (
                outputs if isinstance(outputs, torch.Tensor) else outputs.pooler_output
            )
            img_feats = img_feats / img_feats.norm(p=2, dim=-1, keepdim=True)

        vector = img_feats.to(torch.float32).cpu().numpy()

        # Parametryczny UMAP przyjmuje wektor (1, 512) i zwraca (1, 3)
        # Uwaga: dla standardowego (nie parametrycznego) UMAP transform() może być wolne!
        reduced = self.umap_reducer.transform(vector)

        return float(reduced[0][0]), float(reduced[0][1]), float(reduced[0][2])


ml_service = MLService()
