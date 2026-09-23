"""
Weights Manager for ADVANCE-FER Production Service.
Handles automatic downloading, verification, and caching of pre-trained SOTA ONNX models.
"""

import os
import sys
import logging
import urllib.request
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

DEFAULT_MODEL_URL = "https://huggingface.co/Xenova/facial_emotions_image_detection/resolve/main/onnx/model_quantized.onnx"
DEFAULT_CHECKPOINT_DIR = Path(__file__).resolve().parent / "checkpoints"
DEFAULT_MODEL_PATH = DEFAULT_CHECKPOINT_DIR / "emotion_model.onnx"


def ensure_model_weights(
    model_path: Optional[str] = None,
    download_url: str = DEFAULT_MODEL_URL,
    force_download: bool = False
) -> str:
    """
    Ensures that the pre-trained emotion ONNX model exists locally.
    Downloads the model automatically if missing.

    Args:
        model_path: Optional custom path to model weights.
        download_url: URL to download weights from if missing.
        force_download: Whether to re-download even if the file exists.

    Returns:
        str: Absolute path to the verified model file.
    """
    target_path = Path(model_path) if model_path else DEFAULT_MODEL_PATH

    if target_path.exists() and not force_download:
        if target_path.stat().st_size > 10_000_000:  # Must be > 10MB
            logger.info("Found cached model weights at %s (%d bytes)", target_path, target_path.stat().st_size)
            return str(target_path)
        else:
            logger.warning("Existing checkpoint at %s appears incomplete or corrupted. Re-downloading.", target_path)

    target_path.parent.mkdir(parents=True, exist_ok=True)
    temp_target = target_path.with_suffix(".tmp")

    logger.info("Downloading pre-trained emotion model from %s to %s...", download_url, target_path)
    try:
        def _reporthook(block_num, block_size, total_size):
            if total_size > 0 and block_num % 500 == 0:
                downloaded = block_num * block_size
                percent = min(100.0, (downloaded / total_size) * 100)
                sys.stdout.write(f"\rDownloading model weights: {percent:.1f}% ({downloaded // (1024*1024)}MB / {total_size // (1024*1024)}MB)")
                sys.stdout.flush()

        opener = urllib.request.build_opener()
        opener.addheaders = [('User-Agent', 'ADVANCE-FER/2.0')]
        urllib.request.install_opener(opener)
        urllib.request.urlretrieve(download_url, temp_target, reporthook=_reporthook)
        print()  # Newline after download bar

        if temp_target.stat().st_size < 10_000_000:
            raise ValueError(f"Downloaded model size is suspiciously small: {temp_target.stat().st_size} bytes")

        # Atomic rename
        if target_path.exists():
            target_path.unlink()
        temp_target.rename(target_path)
        logger.info("Successfully downloaded and cached model to %s", target_path)
        return str(target_path)

    except Exception as e:
        if temp_target.exists():
            temp_target.unlink()
        logger.error("Failed to download model weights from %s: %s", download_url, e)
        raise RuntimeError(
            f"Failed to acquire pre-trained model weights. Error: {e}. "
            f"Please download manually from {download_url} and save to {target_path}"
        ) from e


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    path = ensure_model_weights()
    print(f"Verified model weights ready at: {path}")
