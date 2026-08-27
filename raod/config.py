"""Runtime configuration for model weight paths.

The DINOv3 / dino.txt checkpoints are large (several GB) and are not shipped
with the package. They are resolved at runtime from local paths or URLs, with
sensible defaults that can be overridden via environment variables.
"""

from __future__ import annotations

import os
from pathlib import Path

# Default locations for the downloaded checkpoints. Override any of these with
# the corresponding environment variable, or by passing explicit paths to the
# encoder constructor.
DEFAULT_BACKBONE_WEIGHTS = os.environ.get(
    "RAOD_BACKBONE_WEIGHTS",
    str(Path("C:/tmp/dinov3_vitl16_backbone.pth")),
)
DEFAULT_DINOTXT_WEIGHTS = os.environ.get(
    "RAOD_DINOTXT_WEIGHTS",
    str(Path("C:/tmp/dinov3_vitl16_dinotxt.pth")),
)
# BPE vocabulary for the dino.txt text tokenizer (CLIP-style SimpleTokenizer).
DEFAULT_BPE_PATH = os.environ.get(
    "RAOD_BPE_PATH",
    "https://dl.fbaipublicfiles.com/dinov3/thirdparty/bpe_simple_vocab_16e6.txt.gz",
)

# The image backbone is a DINOv3 ViT-L with patch size 16 (1024-dim backbone,
# 2048-dim after the vision head). Text uses the dino.txt TextTransformer and
# projects to the same 2048-dim space.
EMBED_DIM = 2048
PATCH_SIZE = 16
