# tensorrt_llm/inputs/video_input_processor.py
# Video / multi-frame image preprocessing for InternVL-style models.
# Produces pixel tile tensors and num_patches_list compatible with TensorRT-LLM multimodal flow.
#
# Usage:
#   from tensorrt_llm.inputs.video_input_processor import VideoInputProcessor
#   proc = VideoInputProcessor(model_path="path_or_hf_id", tokenizer=tokenizer)
#   pixel_values, num_patches_list = proc.preprocess_video(video_path, num_segments=8, max_num=1)
#   # assign to encoder
#   encoder.input_processor = proc
#   inputs_for_generate = proc.prepare_prompt_inputs_for_generation(
#       question="<image>\\nDescribe the video", pixel_values=pixel_values, num_patches_list=num_patches_list)
#   outputs = encoder.generate(inputs_for_generate)
#
from __future__ import annotations

from typing import List, Tuple, Optional
import math
import itertools

import torch
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from PIL import Image
from decord import VideoReader, cpu
from transformers import AutoProcessor, AutoImageProcessor, PreTrainedTokenizerBase

# Default ImageNet normalization (InternViT feature_extractor.json uses similar mean/std)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(input_size: int):
    transform = T.Compose(
        [
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    return transform


def _generate_target_ratios(min_num: int = 1, max_num: int = 12):
    ratios = []
    for r in range(1, max_num + 1):
        for c in range(1, max_num + 1):
            prod = r * c
            if min_num <= prod <= max_num:
                ratios.append((r, c))
    ratios = sorted(set(ratios), key=lambda x: x[0] * x[1])
    return ratios


def find_closest_aspect_ratio(aspect_ratio: float, target_ratios: List[Tuple[int, int]], width: int, height: int,
                              image_size: int):
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            ratio_prod = ratio[0] * ratio[1]
            if area > 0.5 * (image_size * image_size) * ratio_prod:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image: Image.Image, min_num: int = 1, max_num: int = 12, image_size: int = 448,
                       use_thumbnail: bool = False) -> List[Image.Image]:
    """
    Given a PIL image, resize and split it into tiles of size (image_size, image_size).
    Returns a list of PIL.Image tiles. If use_thumbnail=True and more than one tile, append a thumbnail tile.
    """
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    target_ratios = _generate_target_ratios(min_num=min_num, max_num=max_num)
    rows, cols = find_closest_aspect_ratio(aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    target_width = int(cols * image_size)
    target_height = int(rows * image_size)

    resized_img = image.resize((target_width, target_height), resample=Image.BICUBIC)

    processed_images: List[Image.Image] = []
    for r in range(rows):
        for c in range(cols):
            left = c * image_size
            upper = r * image_size
            right = left + image_size
            lower = upper + image_size
            tile = resized_img.crop((left, upper, right, lower))
            if tile.size != (image_size, image_size):
                tile = tile.resize((image_size, image_size), resample=Image.BICUBIC)
            processed_images.append(tile)

    if use_thumbnail and not (rows == 1 and cols == 1):
        thumbnail = image.resize((image_size, image_size), resample=Image.BICUBIC)
        processed_images.append(thumbnail)

    return processed_images


class VideoInputProcessor:
    """
    Minimal Video/Image InputProcessor for TensorRT-LLM multimodal pipeline.

    Responsibilities:
    - Provide `tokenizer` attribute (if available), so MultimodalEncoder can update tokenizer with processor's tokenizer.
    - Provide helpers:
       * preprocess_image(image_path) -> (pixel_values_tensor, [num_tiles])
       * preprocess_images(list_of_image_paths) -> (pixel_values, num_patches_list)
       * preprocess_video(video_path, bound=None, input_size=448, max_num=1, num_segments=32)
         -> (pixel_values, num_patches_list)
       * prepare_prompt_inputs_for_generation(...) -> lightweight helper producing a list-of-dicts expected by MultimodalEncoder.generate
    """

    def __init__(self,
                 model_path_or_id: Optional[str] = None,
                 tokenizer: Optional[PreTrainedTokenizerBase] = None,
                 image_size: int = 448,
                 max_num_tiles: int = 12,
                 num_segments: int = 32,
                 use_thumbnail: bool = True,
                 local_files_only: bool = False):
        self.model_path_or_id = model_path_or_id
        self.image_size = image_size
        self.max_num_tiles = max_num_tiles
        self.num_segments = num_segments
        self.use_thumbnail = use_thumbnail
        self.local_files_only = local_files_only

        # Provide tokenizer attribute expected by MultimodalEncoder
        self.tokenizer = tokenizer

        # Try to load AutoProcessor / AutoImageProcessor so we match HF preprocessing if available.
        self._processor = None
        if model_path_or_id is not None:
            try:
                self._processor = AutoProcessor.from_pretrained(
                    model_path_or_id, trust_remote_code=True, local_files_only=local_files_only
                )
            except Exception:
                try:
                    self._processor = AutoImageProcessor.from_pretrained(
                        model_path_or_id, trust_remote_code=True, local_files_only=local_files_only
                    )
                except Exception:
                    self._processor = None

    def _tile_transform(self):
        # If HF AutoProcessor is present and provides a normalization, prefer it,
        # otherwise use the default ImageNet normalization.
        if self._processor is not None and hasattr(self._processor, "image_mean"):
            mean = getattr(self._processor, "image_mean", IMAGENET_MEAN)
            std = getattr(self._processor, "image_std", IMAGENET_STD)
            transform = T.Compose([
                T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
                T.Resize((self.image_size, self.image_size), interpolation=InterpolationMode.BICUBIC),
                T.ToTensor(),
                T.Normalize(mean=mean, std=std),
            ])
            return transform
        return build_transform(self.image_size)

    def preprocess_image(self, image_path: str) -> Tuple[torch.Tensor, List[int]]:
        img = Image.open(image_path).convert("RGB")
        tiles = dynamic_preprocess(img, image_size=self.image_size, use_thumbnail=self.use_thumbnail,
                                   max_num=self.max_num_tiles)
        transform = self._tile_transform()
        tensors = [transform(t) for t in tiles]
        pixel_values = torch.stack(tensors)  # shape = (num_tiles, 3, H, W)
        num_patches_list = [pixel_values.size(0)]
        return pixel_values, num_patches_list

    def preprocess_images(self, image_paths: List[str]) -> Tuple[torch.Tensor, List[int]]:
        all_tensors = []
        num_patches_list = []
        transform = self._tile_transform()
        for p in image_paths:
            img = Image.open(p).convert("RGB")
            tiles = dynamic_preprocess(img, image_size=self.image_size, use_thumbnail=self.use_thumbnail,
                                       max_num=self.max_num_tiles)
            tensors = [transform(t) for t in tiles]
            tstack = torch.stack(tensors)
            all_tensors.append(tstack)
            num_patches_list.append(tstack.size(0))
        if len(all_tensors) == 0:
            return torch.empty(0), []
        pixel_values = torch.cat(all_tensors, dim=0)
        return pixel_values, num_patches_list

    def _sample_frame_indices(self, vr: VideoReader, bound: Optional[Tuple[float, float]], num_segments: int):
        max_frame = len(vr) - 1
        fps = float(vr.get_avg_fps())
        if bound:
            start, end = bound[0], bound[1]
        else:
            start, end = -100000, 100000
        start_idx = max(0, round(start * fps))
        end_idx = min(round(end * fps), max_frame)
        seg_size = float(end_idx - start_idx) / num_segments if num_segments > 0 else 1.0
        frame_indices = [
            int(start_idx + (seg_size / 2) + round(seg_size * idx))
            for idx in range(num_segments)
        ]
        frame_indices = [min(max(0, idx), max_frame) for idx in frame_indices]
        return frame_indices

    def preprocess_video(self,
                         video_path: str,
                         bound: Optional[Tuple[float, float]] = None,
                         input_size: Optional[int] = None,
                         max_num: Optional[int] = None,
                         num_segments: Optional[int] = None) -> Tuple[torch.Tensor, List[int]]:
        """
        Samples frames, tiles each frame, and concatenates tiles across frames.
        Returns: (pixel_values: torch.Tensor shape (total_tiles, 3, H, W), num_patches_list: List[int])
        """
        if input_size is None:
            input_size = self.image_size
        if max_num is None:
            max_num = self.max_num_tiles
        if num_segments is None:
            num_segments = self.num_segments

        vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
        frame_indices = self._sample_frame_indices(vr, bound, num_segments)
        transform = self._tile_transform()

        pixel_values_list = []
        num_patches_list = []
        for frame_index in frame_indices:
            arr = vr[frame_index].asnumpy()
            img = Image.fromarray(arr).convert("RGB")
            tiles = dynamic_preprocess(img, image_size=input_size, use_thumbnail=self.use_thumbnail, max_num=max_num)
            tensors = [transform(t) for t in tiles]
            tstack = torch.stack(tensors)
            pixel_values_list.append(tstack)
            num_patches_list.append(tstack.size(0))

        if len(pixel_values_list) == 0:
            return torch.empty(0), []
        pixel_values = torch.cat(pixel_values_list, dim=0)
        return pixel_values, num_patches_list

    def prepare_prompt_inputs_for_generation(self,
                                             prompt_text: str,
                                             pixel_values: Optional[torch.Tensor] = None,
                                             num_patches_list: Optional[List[int]] = None,
                                             extra: Optional[dict] = None) -> List[dict]:
        """
        Prepare a minimal list-of-dicts structure consumable by MultimodalEncoder.generate().

        The exact shape of PromptInputs used inside TRT-LLM may be different in your local checkout;
        this helper produces a minimal dict that many MultimodalEncoder implementations accept:

          {
            "input_text": "<text prompt>",
            "multi_modal_data": {
                "pixel_values": torch.Tensor,   # shape (total_tiles, 3, H, W)
                "num_patches_list": [int, ...]
            },
            "extra": {...}
          }

        If your MultimodalEncoder expects a different schema (e.g., PromptInputs dataclass), adapt accordingly.
        """
        entry = {
            "input_text": prompt_text,
            "multi_modal_data": None,
            "extra": extra or {}
        }
        if pixel_values is not None:
            entry["multi_modal_data"] = {
                "pixel_values": pixel_values,
                "num_patches_list": num_patches_list or [pixel_values.size(0)]
            }
        return [entry]
