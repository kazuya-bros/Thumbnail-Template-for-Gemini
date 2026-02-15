"""
Main inbetween animation pipeline.

Orchestrates segmentation, inbetween generation, and compositing
to produce intermediate frames from two keyframe images.
"""

import cv2
import numpy as np
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

from .segmentation import AnimeSegmenter
from .inbetween_blend import generate_inbetween_blend, generate_inbetween_poisson
from .inbetween_rife import RIFEInterpolator, generate_inbetween_rife
from .inbetween_gemini import GeminiInbetweener, generate_inbetween_gemini


class InbetweenMode(Enum):
    BLEND = "blend"           # Simple alpha blending (fastest, lowest quality)
    POISSON = "poisson"       # Poisson blending (better edges)
    RIFE = "rife"             # RIFE neural interpolation (high quality, local)
    GEMINI = "gemini"         # Gemini API (highest quality for anime, online)
    HYBRID = "hybrid"         # Gemini + mask compositing (recommended)


@dataclass
class InbetweenParams:
    """Parameters for generating a single inbetween frame."""
    eye_ratio: float = 0.5      # 0.0 = fully open, 1.0 = fully closed
    mouth_ratio: float = 0.5    # 0.0 = fully closed, 1.0 = fully open


@dataclass
class AnimationKeyframe:
    """A keyframe in the animation timeline."""
    time: float                 # Time in seconds
    eye_ratio: float = 0.0
    mouth_ratio: float = 0.0


@dataclass
class PipelineConfig:
    """Configuration for the inbetween pipeline."""
    mode: InbetweenMode = InbetweenMode.BLEND
    device: str = "cuda"
    segmentation_model_path: Optional[str] = None
    rife_model_dir: Optional[str] = None
    gemini_api_key: Optional[str] = None
    gemini_model: str = "gemini-2.5-flash-preview-04-17"
    mask_dilate_px: int = 10
    mask_feather_px: int = 15


class InbetweenPipeline:
    """Main pipeline for generating inbetween animation frames."""

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()

        # Lazy-initialized components
        self._segmenter: Optional[AnimeSegmenter] = None
        self._rife: Optional[RIFEInterpolator] = None
        self._gemini: Optional[GeminiInbetweener] = None

        # Cache for segmentation masks
        self._mask_cache: dict[str, np.ndarray] = {}

    @property
    def segmenter(self) -> AnimeSegmenter:
        if self._segmenter is None:
            self._segmenter = AnimeSegmenter(
                model_path=self.config.segmentation_model_path,
                device=self.config.device,
            )
        return self._segmenter

    @property
    def rife(self) -> RIFEInterpolator:
        if self._rife is None:
            self._rife = RIFEInterpolator(
                model_dir=self.config.rife_model_dir,
                device=self.config.device,
            )
        return self._rife

    @property
    def gemini(self) -> GeminiInbetweener:
        if self._gemini is None:
            self._gemini = GeminiInbetweener(
                api_key=self.config.gemini_api_key,
                model=self.config.gemini_model,
            )
        return self._gemini

    def load_image(self, path: str) -> np.ndarray:
        """Load an image from file."""
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(f"Could not load image: {path}")
        # Handle alpha channel
        if img.shape[2] == 4:
            # Convert BGRA to BGR
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        return img

    def get_masks(
        self,
        image_a: np.ndarray,
        image_b: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Get eye and mouth masks from both images.

        Returns:
            (eye_mask, mouth_mask) - binary masks (H, W), uint8
        """
        eye_mask = self.segmenter.get_combined_mask(
            image_a, image_b, "eye", dilate_px=self.config.mask_dilate_px
        )
        mouth_mask = self.segmenter.get_combined_mask(
            image_a, image_b, "mouth", dilate_px=self.config.mask_dilate_px
        )
        return eye_mask, mouth_mask

    def generate_frame(
        self,
        image_a: np.ndarray,
        image_b: np.ndarray,
        params: InbetweenParams,
        eye_mask: Optional[np.ndarray] = None,
        mouth_mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Generate a single inbetween frame.

        Args:
            image_a: Image A - eyes open, mouth closed (BGR, uint8)
            image_b: Image B - eyes closed, mouth open (BGR, uint8)
            params: Inbetween parameters (eye/mouth ratios)
            eye_mask: Pre-computed eye mask (optional, will compute if None)
            mouth_mask: Pre-computed mouth mask (optional, will compute if None)

        Returns:
            Generated inbetween frame (BGR, uint8)
        """
        # Get masks if not provided
        if eye_mask is None or mouth_mask is None:
            eye_mask, mouth_mask = self.get_masks(image_a, image_b)

        mode = self.config.mode

        if mode == InbetweenMode.GEMINI or mode == InbetweenMode.HYBRID:
            return self._generate_gemini(
                image_a, image_b, params, eye_mask, mouth_mask
            )
        elif mode == InbetweenMode.RIFE:
            return self._generate_rife(
                image_a, image_b, params, eye_mask, mouth_mask
            )
        elif mode == InbetweenMode.POISSON:
            return self._generate_poisson(
                image_a, image_b, params, eye_mask, mouth_mask
            )
        else:  # BLEND
            return self._generate_blend(
                image_a, image_b, params, eye_mask, mouth_mask
            )

    def _generate_blend(
        self,
        image_a: np.ndarray,
        image_b: np.ndarray,
        params: InbetweenParams,
        eye_mask: np.ndarray,
        mouth_mask: np.ndarray,
    ) -> np.ndarray:
        """Generate using alpha blending."""
        result = image_a.copy()
        # Blend eyes
        result = generate_inbetween_blend(result, image_b, eye_mask, params.eye_ratio)
        # Blend mouth
        result = generate_inbetween_blend(result, image_b, mouth_mask, params.mouth_ratio)
        return result

    def _generate_poisson(
        self,
        image_a: np.ndarray,
        image_b: np.ndarray,
        params: InbetweenParams,
        eye_mask: np.ndarray,
        mouth_mask: np.ndarray,
    ) -> np.ndarray:
        """Generate using Poisson blending."""
        result = image_a.copy()
        result = generate_inbetween_poisson(result, image_b, eye_mask, params.eye_ratio)
        result = generate_inbetween_poisson(result, image_b, mouth_mask, params.mouth_ratio)
        return result

    def _generate_rife(
        self,
        image_a: np.ndarray,
        image_b: np.ndarray,
        params: InbetweenParams,
        eye_mask: np.ndarray,
        mouth_mask: np.ndarray,
    ) -> np.ndarray:
        """Generate using RIFE interpolation."""
        result = image_a.copy()
        result = generate_inbetween_rife(
            result, image_b, eye_mask, params.eye_ratio, self.rife
        )
        result = generate_inbetween_rife(
            result, image_b, mouth_mask, params.mouth_ratio, self.rife
        )
        return result

    def _generate_gemini(
        self,
        image_a: np.ndarray,
        image_b: np.ndarray,
        params: InbetweenParams,
        eye_mask: np.ndarray,
        mouth_mask: np.ndarray,
    ) -> np.ndarray:
        """Generate using Gemini API with mask compositing (hybrid mode)."""
        # Generate full frame via Gemini
        gemini_result = self.gemini.generate_inbetween(
            image_a, image_b,
            eye_ratio=params.eye_ratio,
            mouth_ratio=params.mouth_ratio,
        )

        if self.config.mode == InbetweenMode.HYBRID:
            # Composite: only use Gemini output within eye+mouth masks
            combined_mask = cv2.bitwise_or(eye_mask, mouth_mask)
            soft_mask = combined_mask.astype(np.float32) / 255.0
            ksize = self.config.mask_feather_px * 2 + 1
            soft_mask = cv2.GaussianBlur(soft_mask, (ksize, ksize), 0)
            soft_mask_3ch = np.stack([soft_mask] * 3, axis=-1)

            result = (
                image_a.astype(np.float32) * (1.0 - soft_mask_3ch)
                + gemini_result.astype(np.float32) * soft_mask_3ch
            )
            return np.clip(result, 0, 255).astype(np.uint8)
        else:
            # Full Gemini output (no masking)
            return gemini_result

    def generate_animation_frames(
        self,
        image_a: np.ndarray,
        image_b: np.ndarray,
        keyframes: list[AnimationKeyframe],
        fps: int = 24,
    ) -> list[np.ndarray]:
        """
        Generate all frames for an animation sequence.

        Args:
            image_a: Image A - eyes open, mouth closed
            image_b: Image B - eyes closed, mouth open
            keyframes: List of animation keyframes with timing
            fps: Target FPS for the animation

        Returns:
            List of frames (BGR, uint8)
        """
        if len(keyframes) < 2:
            raise ValueError("Need at least 2 keyframes")

        # Pre-compute masks (reused for all frames)
        print("[Pipeline] Computing segmentation masks...")
        eye_mask, mouth_mask = self.get_masks(image_a, image_b)

        # Calculate total duration and frame count
        total_duration = keyframes[-1].time - keyframes[0].time
        total_frames = int(total_duration * fps)

        frames = []
        print(f"[Pipeline] Generating {total_frames} frames...")

        for frame_idx in range(total_frames + 1):
            current_time = keyframes[0].time + (frame_idx / fps)

            # Interpolate keyframe parameters at current time
            params = self._interpolate_keyframes(keyframes, current_time)

            frame = self.generate_frame(
                image_a, image_b, params, eye_mask, mouth_mask
            )
            frames.append(frame)

            if (frame_idx + 1) % 10 == 0 or frame_idx == total_frames:
                print(f"  [{frame_idx + 1}/{total_frames + 1}] eye={params.eye_ratio:.2f}, mouth={params.mouth_ratio:.2f}")

        return frames

    def _interpolate_keyframes(
        self,
        keyframes: list[AnimationKeyframe],
        time: float,
    ) -> InbetweenParams:
        """Linearly interpolate between keyframes at the given time."""
        # Find surrounding keyframes
        if time <= keyframes[0].time:
            return InbetweenParams(
                eye_ratio=keyframes[0].eye_ratio,
                mouth_ratio=keyframes[0].mouth_ratio,
            )
        if time >= keyframes[-1].time:
            return InbetweenParams(
                eye_ratio=keyframes[-1].eye_ratio,
                mouth_ratio=keyframes[-1].mouth_ratio,
            )

        for i in range(len(keyframes) - 1):
            if keyframes[i].time <= time <= keyframes[i + 1].time:
                t = (time - keyframes[i].time) / (keyframes[i + 1].time - keyframes[i].time)
                return InbetweenParams(
                    eye_ratio=keyframes[i].eye_ratio + t * (keyframes[i + 1].eye_ratio - keyframes[i].eye_ratio),
                    mouth_ratio=keyframes[i].mouth_ratio + t * (keyframes[i + 1].mouth_ratio - keyframes[i].mouth_ratio),
                )

        return InbetweenParams()


def create_blink_talk_animation(
    steps: int = 4,
    blink_duration: float = 0.15,
    talk_duration: float = 0.3,
) -> list[AnimationKeyframe]:
    """
    Create a standard blink + talk animation keyframe sequence.

    Produces a natural-looking cycle:
    1. Start: eyes open, mouth closed
    2. Begin talking (mouth opens)
    3. Blink while talking
    4. End talking (mouth closes)
    5. Return to start

    Args:
        steps: Number of intermediate steps for each transition
        blink_duration: Duration of blink in seconds
        talk_duration: Duration of talk cycle in seconds

    Returns:
        List of AnimationKeyframes
    """
    keyframes = [
        # Start: neutral
        AnimationKeyframe(time=0.0, eye_ratio=0.0, mouth_ratio=0.0),
        # Begin talking
        AnimationKeyframe(time=0.1, eye_ratio=0.0, mouth_ratio=0.5),
        AnimationKeyframe(time=0.2, eye_ratio=0.0, mouth_ratio=1.0),
        # Blink while talking
        AnimationKeyframe(time=0.35, eye_ratio=0.5, mouth_ratio=0.7),
        AnimationKeyframe(time=0.45, eye_ratio=1.0, mouth_ratio=0.5),
        AnimationKeyframe(time=0.55, eye_ratio=0.5, mouth_ratio=0.7),
        AnimationKeyframe(time=0.65, eye_ratio=0.0, mouth_ratio=1.0),
        # End talking
        AnimationKeyframe(time=0.8, eye_ratio=0.0, mouth_ratio=0.5),
        AnimationKeyframe(time=0.95, eye_ratio=0.0, mouth_ratio=0.0),
        # Hold neutral
        AnimationKeyframe(time=1.2, eye_ratio=0.0, mouth_ratio=0.0),
    ]
    return keyframes
