"""
Alpha-blend based inbetween generation.

The simplest and fastest approach: directly blend pixel values
between two images within masked regions.
"""

import cv2
import numpy as np


def generate_inbetween_blend(
    image_a: np.ndarray,
    image_b: np.ndarray,
    mask: np.ndarray,
    ratio: float,
) -> np.ndarray:
    """
    Generate an inbetween frame by alpha-blending within a masked region.

    Args:
        image_a: Source image (BGR, uint8)
        image_b: Target image (BGR, uint8)
        mask: Binary mask defining the blend region (H, W), uint8, 0 or 255
        ratio: Blend ratio. 0.0 = image_a, 1.0 = image_b

    Returns:
        Blended image (BGR, uint8)
    """
    ratio = np.clip(ratio, 0.0, 1.0)

    # Create a soft mask with feathered edges for smoother blending
    soft_mask = create_soft_mask(mask, feather_px=15)

    # Alpha blend within the masked region
    mask_3ch = np.stack([soft_mask] * 3, axis=-1)
    blended_region = (
        image_a.astype(np.float32) * (1.0 - ratio)
        + image_b.astype(np.float32) * ratio
    )

    # Composite: use blended region where mask is active, original elsewhere
    result = (
        image_a.astype(np.float32) * (1.0 - mask_3ch)
        + blended_region * mask_3ch
    )

    return np.clip(result, 0, 255).astype(np.uint8)


def generate_inbetween_poisson(
    image_a: np.ndarray,
    image_b: np.ndarray,
    mask: np.ndarray,
    ratio: float,
) -> np.ndarray:
    """
    Generate an inbetween frame using Poisson blending (seamless clone).
    Produces more natural results than simple alpha blending.

    Args:
        image_a: Source/base image (BGR, uint8)
        image_b: Target image (BGR, uint8)
        mask: Binary mask defining the blend region (H, W), uint8, 0 or 255
        ratio: Blend ratio. 0.0 = image_a, 1.0 = image_b

    Returns:
        Composited image (BGR, uint8)
    """
    ratio = np.clip(ratio, 0.0, 1.0)

    if ratio < 0.01:
        return image_a.copy()
    if ratio > 0.99:
        return image_b.copy()

    # First, create the blended source (what we want to paste)
    blended = cv2.addWeighted(image_a, 1.0 - ratio, image_b, ratio, 0)

    # Find the center of the mask for seamlessClone
    moments = cv2.moments(mask)
    if moments["m00"] == 0:
        return image_a.copy()

    cx = int(moments["m10"] / moments["m00"])
    cy = int(moments["m01"] / moments["m00"])

    # Ensure mask is proper format for seamlessClone
    mask_clean = (mask > 127).astype(np.uint8) * 255

    try:
        result = cv2.seamlessClone(
            blended, image_a, mask_clean, (cx, cy), cv2.NORMAL_CLONE
        )
        return result
    except cv2.error:
        # Fallback to simple blend if seamlessClone fails
        return generate_inbetween_blend(image_a, image_b, mask, ratio)


def create_soft_mask(mask: np.ndarray, feather_px: int = 15) -> np.ndarray:
    """
    Create a soft (feathered) mask from a binary mask.

    Args:
        mask: Binary mask (H, W), uint8, 0 or 255
        feather_px: Gaussian blur radius for feathering

    Returns:
        Soft mask (H, W), float32, range [0, 1]
    """
    soft = mask.astype(np.float32) / 255.0
    if feather_px > 0:
        ksize = feather_px * 2 + 1
        soft = cv2.GaussianBlur(soft, (ksize, ksize), 0)
    return soft
