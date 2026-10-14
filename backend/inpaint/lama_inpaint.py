import cv2
import torch
import numpy as np
from typing import Union, List
from PIL import Image
from backend.inpaint.utils.lama_util import prepare_img_and_mask, get_image, pad_img_to_modulo


class LamaInpaint:
    def __init__(self, device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"), model_path='big-lama.pt') -> None:
        self.model = torch.jit.load(model_path, map_location=device)
        self.model.eval()
        self.device = device

    def inpaint(self, image: Union[Image.Image, np.ndarray], mask: Union[Image.Image, np.ndarray]):
        if isinstance(image, np.ndarray):
            orig_height, orig_width = image.shape[:2]
        else:
            orig_height, orig_width = np.array(image).shape[:2]
        image, mask = prepare_img_and_mask(image, mask, self.device)
        with torch.inference_mode():
            inpainted = self.model(image, mask)
            cur_res = inpainted[0].permute(1, 2, 0).detach().cpu().numpy()
            cur_res = np.clip(cur_res * 255, 0, 255).astype('uint8')
            cur_res = cur_res[:orig_height, :orig_width]
            return cur_res

    def _inpaint_batch(self, images: List[np.ndarray], masks: List[np.ndarray]):
        """批量推理：将多帧分小批次送入 GPU，避免单次推理过大导致卡死"""
        if len(images) == 1:
            return [self.inpaint(images[0], masks[0])]

        orig_height, orig_width = images[0].shape[:2]
        # 分小批次推理。小裁剪可以多帧一起跑，大图保持小批次。
        budget = 512 * 512 * 4
        mini_batch_size = max(1, min(8, budget // max(orig_height * orig_width, 1)))
        results = [None] * len(images)
        for start in range(0, len(images), mini_batch_size):
            end = min(start + mini_batch_size, len(images))
            batch_imgs = []
            batch_masks = []
            for i in range(start, end):
                batch_imgs.append(get_image(images[i]))
                batch_masks.append(get_image(masks[i]))

            padded_imgs = np.stack([pad_img_to_modulo(img, 8) for img in batch_imgs])
            padded_masks = np.stack([pad_img_to_modulo(m, 8) for m in batch_masks])

            img_tensor = torch.from_numpy(padded_imgs).to(self.device)
            mask_tensor = torch.from_numpy(padded_masks).to(self.device)
            mask_tensor = (mask_tensor > 0) * 1

            with torch.inference_mode():
                inpainted = self.model(img_tensor, mask_tensor)
                batch_results = inpainted.permute(0, 2, 3, 1).detach().cpu().numpy()
                batch_results = np.clip(batch_results * 255, 0, 255).astype('uint8')

            for i in range(end - start):
                results[start + i] = batch_results[i][:orig_height, :orig_width]

            del img_tensor, mask_tensor, padded_imgs, padded_masks
            if torch.cuda.is_available() and orig_height * orig_width * mini_batch_size > budget:
                torch.cuda.empty_cache()

        return results

    def _prepare_crop(self, image: np.ndarray, mask: np.ndarray):
        """Keep BGR. This checkpoint is called with OpenCV frames elsewhere in the repo."""
        height, width = image.shape[:2]
        long_side = max(height, width)
        max_side = 768
        if long_side > max_side:
            scale = max_side / float(long_side)
            resized = (max(8, int(round(width * scale))), max(8, int(round(height * scale))))
            image = cv2.resize(image, resized, interpolation=cv2.INTER_AREA)
            mask = cv2.resize(mask, resized, interpolation=cv2.INTER_NEAREST)
        return image, mask

    def _inpaint_crops(self, images: List[np.ndarray], mask: np.ndarray):
        """Inpaint BGR crops that share one mask. Returns BGR at the original crop size."""
        height, width = images[0].shape[:2]
        prepared = [self._prepare_crop(image, mask) for image in images]
        batch_images = [item[0] for item in prepared]
        batch_masks = [item[1] for item in prepared]
        preds = self._inpaint_batch(batch_images, batch_masks)
        restored = []
        for pred in preds:
            if pred.shape[1] != width or pred.shape[0] != height:
                pred = cv2.resize(pred, (width, height), interpolation=cv2.INTER_CUBIC)
            restored.append(pred)
        return restored

    def __call__(self, input_frames: List[np.ndarray], input_mask: np.ndarray):
        """
        Crop each subtitle island, inpaint that crop with LaMa, and write back
        only the feathered mask. Pixels outside the mask stay the original frame.
        """
        if not input_frames:
            return []
        mask2d = input_mask[:, :, 0] if input_mask.ndim == 3 else input_mask
        binary = (mask2d > 0).astype(np.uint8)
        if not np.any(binary):
            return list(input_frames)

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary * 255, connectivity=8)
        components = []
        for label in range(1, num_labels):
            x, y, width, height, area = (int(v) for v in stats[label])
            if area < 10 or width < 2 or height < 2:
                continue
            components.append((area, label, x, y, width, height))
        if not components:
            return list(input_frames)

        frames = [frame.copy() for frame in input_frames]
        frame_h, frame_w = binary.shape[:2]
        # Larger holes first so a later small box sees the finished neighbor as context.
        for area, label, x, y, width, height in sorted(components, reverse=True):
            pad = max(64, min(160, max(width, height)))
            x0 = max(0, x - pad)
            y0 = max(0, y - pad)
            x1 = min(frame_w, x + width + pad)
            y1 = min(frame_h, y + height + pad)
            hole = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
            hole[labels[y0:y1, x0:x1] == label] = 255
            if not np.any(hole):
                continue
            crops = [frame[y0:y1, x0:x1].copy() for frame in frames]
            preds = self._inpaint_crops(crops, hole)
            # Soft edge only. The interior of the box is the LaMa fill.
            alpha = cv2.GaussianBlur(hole, (0, 0), 2.0).astype(np.float32) / 255.0
            alpha = alpha[:, :, None]
            print(
                f"LaMa crop {x1 - x0}x{y1 - y0} at ({x0},{y0}) hole {width}x{height} frames={len(frames)}",
                flush=True,
            )
            for index, pred in enumerate(preds):
                region = frames[index][y0:y1, x0:x1]
                if pred.shape[:2] != region.shape[:2]:
                    pred = cv2.resize(pred, (region.shape[1], region.shape[0]), interpolation=cv2.INTER_CUBIC)
                blended = pred.astype(np.float32) * alpha + region.astype(np.float32) * (1.0 - alpha)
                frames[index][y0:y1, x0:x1] = np.clip(blended, 0, 255).astype(np.uint8)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return frames
