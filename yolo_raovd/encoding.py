from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


def _normalize(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    norm = np.linalg.norm(v)
    if norm < eps:
        return v
    return v / norm


def _to_numpy(x: Any) -> np.ndarray:
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float32)


def _array_from_pil(image_path: str) -> np.ndarray:
    from PIL import Image

    with Image.open(image_path) as im:
        return np.array(im.convert("RGB"), dtype=np.float32)


def _array_to_pil(image: np.ndarray):
    from PIL import Image

    arr = np.asarray(image)
    if arr.ndim == 2:
        arr = np.repeat(arr[:, :, None], 3, axis=2)
    if arr.shape[-1] == 1:
        arr = np.repeat(arr[:, :, :1], 3, axis=2)
    if arr.shape[-1] > 3:
        arr = arr[:, :, :3]
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


@dataclass
class SimpleTextEncoder:
    """CLIP text encoder used for text prompts."""

    model_name: str = "openai/clip-vit-base-patch32"
    device: str = "cpu"
    _resources: Dict[str, Any] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        self._lock = threading.Lock()

    @property
    def dim(self) -> int:
        self._load()
        return int(self._resources.get("dim", 512))

    def _load(self) -> None:
        if self._resources:
            return
        with self._lock:
            if self._resources:
                return
            try:
                from transformers import CLIPModel, CLIPProcessor
            except Exception as exc:  # pragma: no cover - optional dependency guard
                raise RuntimeError(
                    "transformers is required for SimpleTextEncoder; install with `pip install transformers torch`."
                ) from exc

            model = CLIPModel.from_pretrained(self.model_name)
            processor = CLIPProcessor.from_pretrained(self.model_name)
            model.to(self.device)
            model.eval()
            dim = int(getattr(model.config, "projection_dim", 512))
            self._resources.update({"model": model, "processor": processor, "dim": dim})

    def encode(self, text: str) -> np.ndarray:
        self._load()
        if not isinstance(text, str) or not text.strip():
            return np.zeros(int(self._resources.get("dim", 512)), dtype=np.float32)

        inputs = self._resources["processor"](
            text=[text],
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with np.errstate(all="ignore"):
            feats = self._resources["model"].get_text_features(**inputs)
        if hasattr(feats, "pooler_output") and feats.pooler_output is not None:
            feat = _to_numpy(feats.pooler_output)
        elif hasattr(feats, "last_hidden_state") and feats.last_hidden_state is not None:
            feat = _to_numpy(feats.last_hidden_state)
            if feat.ndim == 3:
                feat = feat[0]
            feat = feat.mean(axis=0)
        else:
            feat = _to_numpy(feats)
            if feat.ndim == 3:
                feat = feat[0]
            if feat.ndim > 1:
                feat = feat[0]
        return _normalize(feat.astype(np.float32)).astype(np.float32)


@dataclass
class SimpleCLIPImageEncoder:
    """CLIP image encoder for text-query branch (CLIP-aligned image embeddings)."""

    model_name: str = "openai/clip-vit-base-patch32"
    device: str = "cpu"
    _resources: Dict[str, Any] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        self._lock = threading.Lock()

    @property
    def dim(self) -> int:
        self._load()
        return int(self._resources.get("dim", 512))

    def _load(self) -> None:
        if self._resources:
            return
        with self._lock:
            if self._resources:
                return
            try:
                from transformers import CLIPModel, CLIPProcessor
            except Exception as exc:  # pragma: no cover - optional dependency guard
                raise RuntimeError(
                    "transformers is required for SimpleCLIPImageEncoder; install with `pip install transformers torch`."
                ) from exc

            model = CLIPModel.from_pretrained(self.model_name)
            processor = CLIPProcessor.from_pretrained(self.model_name)
            model.to(self.device)
            model.eval()
            dim = int(getattr(model.config, "projection_dim", 512))
            self._resources.update({"model": model, "processor": processor, "dim": dim})

    def encode(self, image: np.ndarray) -> np.ndarray:
        self._load()
        img = _array_to_pil(image)
        inputs = self._resources["processor"](images=img, return_tensors="pt")
        if hasattr(next(iter(inputs.values())), "to"):
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with np.errstate(all="ignore"):
            feats = self._resources["model"].get_image_features(**inputs)

        if hasattr(feats, "pooler_output") and feats.pooler_output is not None:
            feat = _to_numpy(feats.pooler_output)
        elif hasattr(feats, "last_hidden_state") and feats.last_hidden_state is not None:
            feat = _to_numpy(feats.last_hidden_state)
            if feat.ndim == 3:
                feat = feat[0]
            if feat.ndim > 1:
                feat = feat.mean(axis=0)
        else:
            feat = _to_numpy(feats)
            if feat.ndim == 3:
                feat = feat[0]
            if feat.ndim > 1:
                feat = feat.mean(axis=0)
        vector = np.asarray(feat, dtype=np.float32)
        if vector.ndim > 1:
            vector = vector.reshape(-1)
        return _normalize(vector).astype(np.float32)

    def encode_image(self, image_path: str) -> np.ndarray:
        return self.encode(_array_from_pil(image_path))


@dataclass
class SimpleImageEncoder:
    """DINOv3 image encoder used for region embeddings and image-query matching."""

    model_name: str = "facebook/dinov3-vits16plus-pretrain-lvd1689m"
    fallback_model_names: Optional[List[str]] = None
    device: str = "cpu"
    _resources: Dict[str, Any] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        self._lock = threading.Lock()
        if self.fallback_model_names is None:
            self.fallback_model_names = [
                "facebook/dinov3-vits16plus-pretrain-lvd1689m",
                "facebook/dinov3-vit7b16-pretrain-lvd1689m",
                "facebook/dinov2-base",
            ]

    @property
    def dim(self) -> int:
        self._load()
        return int(self._resources.get("dim", 768))

    def _load(self) -> None:
        if self._resources:
            return
        with self._lock:
            if self._resources:
                return
            try:
                from transformers import AutoImageProcessor, AutoModel
            except Exception as exc:  # pragma: no cover - optional dependency guard
                raise RuntimeError(
                    "transformers is required for SimpleImageEncoder; install with `pip install transformers torch`."
                ) from exc

            candidates = [self.model_name] + [m for m in (self.fallback_model_names or []) if m != self.model_name]
            last_err: Optional[Exception] = None
            for model_name in candidates:
                try:
                    processor = AutoImageProcessor.from_pretrained(model_name)
                    model = AutoModel.from_pretrained(model_name)
                    model.to(self.device)
                    model.eval()
                    dim = int(getattr(model.config, "hidden_size", 768))
                    self._resources.update({"model": model, "processor": processor, "dim": dim, "model_name": model_name})
                    return
                except Exception as exc:  # pragma: no cover - runtime guard
                    last_err = exc
                    self._resources = {}
            raise RuntimeError(
                f"failed to load DINOv3 model names: {candidates}"
            ) from last_err

    def _extract_feature(self, pixel_values) -> np.ndarray:
        outputs = self._resources["model"](pixel_values=pixel_values)
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            feat = outputs.pooler_output
        else:
            hidden = outputs.last_hidden_state
            if hidden.ndim != 3 or hidden.shape[1] <= 1:
                feat = hidden[:, 0]
            else:
                feat = hidden[:, 1:].mean(axis=1)
        return _normalize(_to_numpy(feat[0]))

    def encode(self, image: np.ndarray) -> np.ndarray:
        self._load()
        img = _array_to_pil(image)
        inputs = self._resources["processor"](images=img, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device)
        return self._extract_feature(pixel_values)

    def encode_image(self, image_path: str) -> np.ndarray:
        return self.encode(_array_from_pil(image_path))


def stack_embeddings(vectors: List[np.ndarray]) -> np.ndarray:
    if len(vectors) == 0:
        return np.empty((0, 0), dtype=np.float32)
    return np.stack(vectors, axis=0).astype(np.float32)
