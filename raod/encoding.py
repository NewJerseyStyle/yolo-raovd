from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

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
class DINOV3Encoder:
    """Unified image + text encoder built on DINOv3 and dino.txt.

    Both images and text are projected into a shared 2048-dimensional space:

    * Images: DINOv3 ViT-L backbone + a 2-block vision head -> 2048-dim.
    * Text: dino.txt TextTransformer + a linear projection -> 2048-dim.

    ``encode_image`` returns a single pooled vector (for reference-store rows and
    whole-image queries). ``encode_image_spatial`` returns per-patch 2048-dim
    features (for dense tiling matching). ``encode_text`` returns a 2048-dim
    vector for text queries/rows.
    """

    backbone_weights: str = "C:/tmp/dinov3_vitl16_backbone.pth"
    dinotxt_weights: str = "C:/tmp/dinov3_vitl16_dinotxt.pth"
    bpe_path_or_url: str = "https://dl.fbaipublicfiles.com/dinov3/thirdparty/bpe_simple_vocab_16e6.txt.gz"
    device: str = "cpu"
    img_size: int = 224
    _resources: Dict[str, Any] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        self._lock = threading.Lock()

    @property
    def dim(self) -> int:
        return 2048

    def _load(self) -> None:
        if self._resources:
            return
        with self._lock:
            if self._resources:
                return
            # Make the vendored dinov3 package importable.
            import raod._vendor  # noqa: F401
            from dinov3.hub.dinotxt import dinov3_vitl16_dinotxt_tet1280d20h24l

            model, tokenizer = dinov3_vitl16_dinotxt_tet1280d20h24l(
                pretrained=True,
                weights=self.dinotxt_weights,
                backbone_weights=self.backbone_weights,
                bpe_path_or_url=self.bpe_path_or_url,
            )
            model.to(self.device)
            model.eval()
            self._resources.update({"model": model, "tokenizer": tokenizer})

    def _preprocess(self, image: np.ndarray) -> Any:
        import torch
        import torchvision.transforms as T

        pil = _array_to_pil(image)
        tf = T.Compose(
            [
                T.Resize((self.img_size, self.img_size)),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        return tf(pil).unsqueeze(0).to(self.device)

    def encode_image(self, image: np.ndarray) -> np.ndarray:
        """Return a single pooled 2048-dim vector for an image."""
        self._load()
        import torch

        tensor = self._preprocess(image)
        with torch.no_grad():
            feat = self._resources["model"].encode_image(tensor, normalize=True)
        return _to_numpy(feat[0]).astype(np.float32)

    def encode_image_spatial(self, image: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int]]:
        """Return per-patch 2048-dim features and the (H, W) patch grid size.

        The returned array has shape (H * W, 2048) where each row is the aligned
        feature for one image patch. The pooled image feature is a concatenation
        of the class token and the mean patch token; to keep each location in the
        same 2048-dim space as text, each patch token is concatenated with the
        (global) class token. This is used for dense tiling matching.
        """
        self._load()
        import torch

        tensor = self._preprocess(image)
        with torch.no_grad():
            pooled, patch_tokens, _ = self._resources["model"].encode_image_with_patch_tokens(
                tensor, normalize=True
            )
        pooled = _to_numpy(pooled[0])  # (2048,) = [class_token(1024), mean_patch(1024)]
        patch = _to_numpy(patch_tokens[0])  # (H*W, 1024)
        class_token = pooled[: patch.shape[1]]  # (1024,)
        spatial = np.concatenate(
            [np.broadcast_to(class_token, (patch.shape[0], class_token.shape[0])), patch],
            axis=1,
        )  # (H*W, 2048)
        grid = int(round(np.sqrt(patch.shape[0])))
        return spatial.astype(np.float32), (grid, grid)

    def encode_batch(self, images: List[np.ndarray]) -> np.ndarray:
        """Return a pooled 2048-dim vector per image (for reference-store building)."""
        self._load()
        import torch

        if not images:
            return np.empty((0, self.dim), dtype=np.float32)
        tensors = torch.cat([self._preprocess(img) for img in images], dim=0)
        with torch.no_grad():
            feats = self._resources["model"].encode_image(tensors, normalize=True)
        return _to_numpy(feats).astype(np.float32)

    def encode_image_path(self, image_path: str) -> np.ndarray:
        return self.encode_image(_array_from_pil(image_path))

    def encode_text(self, text: str) -> np.ndarray:
        """Return a 2048-dim vector for a text query/row."""
        self._load()
        import torch

        if not isinstance(text, str) or not text.strip():
            return np.zeros(self.dim, dtype=np.float32)
        tokens = self._resources["tokenizer"].tokenize([text], context_length=77).to(self.device)
        with torch.no_grad():
            feat = self._resources["model"].encode_text(tokens, normalize=True)
        return _to_numpy(feat[0]).astype(np.float32)

    def encode_text_batch(self, texts: List[str]) -> np.ndarray:
        self._load()
        import torch

        if not texts:
            return np.empty((0, self.dim), dtype=np.float32)
        tokens = self._resources["tokenizer"].tokenize(texts, context_length=77).to(self.device)
        with torch.no_grad():
            feats = self._resources["model"].encode_text(tokens, normalize=True)
        return _to_numpy(feats).astype(np.float32)


def stack_embeddings(vectors: List[np.ndarray]) -> np.ndarray:
    if len(vectors) == 0:
        return np.empty((0, 0), dtype=np.float32)
    return np.stack(vectors, axis=0).astype(np.float32)
