"""STTN inference helpers.

Keep the window loop on GPU. One transfer back to NumPy after the clip,
instead of a sync on every neighbor stride.
"""
import cv2
import numpy as np
import torch


def enable_cuda_fast_math(device):
    """Turn on cuDNN autotune and TF32. Returns whether CUDA autocast should run."""
    if getattr(device, "type", None) != "cuda":
        return False
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    return True


def bgr_frames_to_tensor(frames):
    """(T, 3, H, W) float RGB in [0, 1].

    Matches Stack (BGR to RGB) plus ToTorchFormatTensor, without the PIL copy.
    """
    height, width = frames[0].shape[:2]
    packed = np.empty((len(frames), height, width, 3), dtype=np.uint8)
    for i, frame in enumerate(frames):
        packed[i] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return torch.from_numpy(packed).permute(0, 3, 1, 2).contiguous().float().div(255)


def windows_to_uint8(acc):
    """EMA-blended predictions (T, 3, H, W) in [0, 1] to uint8 HWC."""
    return (
        torch.stack(acc)
        .permute(0, 2, 3, 1)
        .clamp_(0, 1)
        .mul_(255)
        .to(dtype=torch.uint8)
        .cpu()
        .numpy()
    )
