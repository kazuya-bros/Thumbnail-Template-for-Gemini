"""
Gemini API (Nano Banana) based inbetween generation.

Uses Google's Gemini 2.5 Flash Image model to generate semantically correct
intermediate frames between two anime illustrations.
"""

import cv2
import numpy as np
import base64
import io
from pathlib import Path
from typing import Optional


class GeminiInbetweener:
    """Generate inbetween frames using Gemini API."""

    def __init__(self, api_key: Optional[str] = None, model: str = "gemini-2.5-flash-preview-04-17"):
        self.api_key = api_key
        self.model_name = model
        self.client = None
        self._initialized = False

    def _init_client(self):
        """Initialize the Gemini API client."""
        if self._initialized:
            return

        if not self.api_key:
            import os
            self.api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")

        if not self.api_key:
            raise ValueError(
                "Gemini API key not provided. Set GEMINI_API_KEY or GOOGLE_API_KEY environment variable, "
                "or pass api_key to constructor."
            )

        try:
            from google import genai
            self.client = genai.Client(api_key=self.api_key)
            self._initialized = True
            print(f"[Gemini] Initialized with model: {self.model_name}")
        except ImportError:
            raise ImportError("google-genai package not installed. Run: pip install google-genai")

    def generate_inbetween(
        self,
        image_a: np.ndarray,
        image_b: np.ndarray,
        eye_ratio: float = 0.5,
        mouth_ratio: float = 0.5,
    ) -> np.ndarray:
        """
        Generate an inbetween frame using Gemini API.

        Args:
            image_a: Image A - eyes open, mouth closed (BGR, uint8)
            image_b: Image B - eyes closed, mouth open (BGR, uint8)
            eye_ratio: Eye closure ratio. 0.0 = fully open, 1.0 = fully closed
            mouth_ratio: Mouth opening ratio. 0.0 = fully closed, 1.0 = fully open

        Returns:
            Generated inbetween image (BGR, uint8)
        """
        self._init_client()
        from google.genai import types

        # Convert images to PIL for the API
        from PIL import Image

        img_a_rgb = cv2.cvtColor(image_a, cv2.COLOR_BGR2RGB)
        img_b_rgb = cv2.cvtColor(image_b, cv2.COLOR_BGR2RGB)
        pil_a = Image.fromarray(img_a_rgb)
        pil_b = Image.fromarray(img_b_rgb)

        eye_pct = int(eye_ratio * 100)
        mouth_pct = int(mouth_ratio * 100)

        prompt = (
            f"I have two illustrations of the exact same anime character in the exact same pose and setting.\n"
            f"Image 1: The character's eyes are FULLY OPEN and mouth is FULLY CLOSED.\n"
            f"Image 2: The character's eyes are FULLY CLOSED and mouth is FULLY OPEN.\n\n"
            f"Generate a NEW illustration that is the intermediate state between these two images:\n"
            f"- Eyes should be {eye_pct}% closed (0% = fully open like Image 1, 100% = fully closed like Image 2)\n"
            f"- Mouth should be {mouth_pct}% open (0% = fully closed like Image 1, 100% = fully open like Image 2)\n\n"
            f"CRITICAL REQUIREMENTS:\n"
            f"- Keep the EXACT same art style, line quality, coloring, and shading\n"
            f"- Keep the EXACT same pose, clothing, hair, accessories, and background\n"
            f"- ONLY change the eyes and mouth to the specified intermediate state\n"
            f"- The result must look like it was drawn by the same artist\n"
            f"- Output at the same resolution as the input images"
        )

        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=[prompt, pil_a, pil_b],
                config=types.GenerateContentConfig(
                    response_modalities=["IMAGE", "TEXT"],
                ),
            )

            # Extract image from response
            for part in response.candidates[0].content.parts:
                if part.inline_data and part.inline_data.mime_type.startswith("image/"):
                    image_bytes = part.inline_data.data
                    nparr = np.frombuffer(image_bytes, np.uint8)
                    result = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    if result is not None:
                        # Resize to match input if needed
                        h, w = image_a.shape[:2]
                        if result.shape[:2] != (h, w):
                            result = cv2.resize(result, (w, h), interpolation=cv2.INTER_LANCZOS4)
                        return result

            print("[Gemini] No image in response. Response text:")
            for part in response.candidates[0].content.parts:
                if part.text:
                    print(f"  {part.text[:200]}")
            raise RuntimeError("Gemini API did not return an image")

        except Exception as e:
            print(f"[Gemini] API call failed: {e}")
            raise


def generate_inbetween_gemini(
    image_a: np.ndarray,
    image_b: np.ndarray,
    mask: np.ndarray,
    eye_ratio: float = 0.5,
    mouth_ratio: float = 0.5,
    api_key: Optional[str] = None,
    inbetweener: Optional[GeminiInbetweener] = None,
) -> np.ndarray:
    """
    Generate an inbetween frame using Gemini, composited with mask.
    Only the masked region from Gemini output is used; the rest comes from image_a.

    Args:
        image_a: Source image (BGR, uint8)
        image_b: Target image (BGR, uint8)
        mask: Combined mask for eye+mouth regions (H, W), uint8
        eye_ratio: Eye closure ratio
        mouth_ratio: Mouth opening ratio
        api_key: Gemini API key (optional if env var is set)
        inbetweener: Reusable GeminiInbetweener instance

    Returns:
        Composited image (BGR, uint8)
    """
    if inbetweener is None:
        inbetweener = GeminiInbetweener(api_key=api_key)

    # Generate the full inbetween via Gemini
    gemini_result = inbetweener.generate_inbetween(image_a, image_b, eye_ratio, mouth_ratio)

    # Composite: use Gemini result only within mask, keep original elsewhere
    soft_mask = mask.astype(np.float32) / 255.0
    ksize = 21
    soft_mask = cv2.GaussianBlur(soft_mask, (ksize, ksize), 0)
    soft_mask_3ch = np.stack([soft_mask] * 3, axis=-1)

    result = (
        image_a.astype(np.float32) * (1.0 - soft_mask_3ch)
        + gemini_result.astype(np.float32) * soft_mask_3ch
    )

    return np.clip(result, 0, 255).astype(np.uint8)
