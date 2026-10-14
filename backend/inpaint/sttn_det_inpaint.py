import time

import cv2
import numpy as np
import torch
from torchvision import transforms
from typing import List
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from backend.config import config
from backend.inpaint.sttn.network_sttn import InpaintGenerator
from backend.inpaint.utils.sttn_utils import Stack, ToTorchFormatTensor
from backend.tools.inpaint_tools import get_inpaint_area_by_mask

# 定义图像预处理方式
_to_tensors = transforms.Compose([
    Stack(),  # 将图像堆叠为序列
    ToTorchFormatTensor()  # 将堆叠的图像转化为PyTorch张量
])

class STTNDetInpaint:
    def __init__(self, device, model_path):
        self.device = device
        # 1. 创建InpaintGenerator模型实例并装载到选择的设备上
        self.model = InpaintGenerator().to(self.device)
        # 2. 载入预训练模型的权重，转载模型的状态字典
        self.model.load_state_dict(torch.load(model_path, map_location='cpu')['netG'])
        # 3. # 将模型设置为评估模式
        self.model.eval()
        # 模型输入用的宽和高
        self.model_input_width, self.model_input_height = 432, 240
        # 2. 设置相连帧数
        self.neighbor_stride = config.sttnNeighborStride.value
        self.ref_length = config.sttnReferenceLength.value

    def __call__(self, input_frames: List[np.ndarray], input_mask: np.ndarray):
        """
        :param input_frames: 原视频帧
        :param mask: 字幕区域mask
        """
        mask = input_mask[:, :, None]
        H_ori, W_ori = mask.shape[:2]
        H_ori = int(H_ori + 0.5)
        W_ori = int(W_ori + 0.5)
        # 扁长横条且周围是横向线条时，用左右像素把每一行接上。
        # 按连通块判断，避免一个小块把整条字幕又送回低分辨率模型。
        if input_frames:
            row_plane = self._row_component_mask(input_frames, mask)
            if np.any(row_plane):
                input_frames = [self._fill_rows_frame(frame, row_plane) for frame in input_frames]
                mask = mask.copy()
                mask[row_plane > 0, :] = 0
            if not np.any(mask):
                return input_frames
        # 确定去字幕的垂直高度部分
        if H_ori > W_ori:
            split_h = int(H_ori * 5 / 9)
        else:
            split_h = int(W_ori * 5 / 18)
        inpaint_area = get_inpaint_area_by_mask(W_ori, H_ori, split_h, mask)
        # 模型画布只有 240 像素高。贴着字幕裁，字才不会被缩成十几像素后再放大成马赛克。
        inpaint_area = self._tighten_areas(inpaint_area, mask)
        # 初始化帧存储变量
        # 高分辨率帧存储列表（浅拷贝 + 逐帧 copy，避免 deepcopy 开销）
        frames_hr = [f.copy() for f in input_frames]
        frames_scaled = {}  # 存放缩放后帧的字典
        masks_scaled = {}  # 存放缩放后遮罩的字典
        comps = {}  # 存放补全后帧的字典
        # 存储最终的视频帧
        inpainted_frames = []
        for k in range(len(inpaint_area)):
            frames_scaled[k] = []  # 为每个去除部分初始化一个列表
            masks_scaled[k] = []  # 为每个去除部分初始化一个列表

        # 读取并缩放帧
        for j in range(len(frames_hr)):
            image = frames_hr[j]
            # 对每个去除部分进行切割和缩放
            for k in range(len(inpaint_area)):
                y0, y1, x0, x1 = inpaint_area[k]
                image_crop = image[y0:y1, x0:x1, :]
                mask_crop = mask[y0:y1, x0:x1, :]
                image_resize = cv2.resize(image_crop, (self.model_input_width, self.model_input_height))  # 缩放
                mask_resize = cv2.resize(mask_crop, (self.model_input_width, self.model_input_height))  # 缩放
                frames_scaled[k].append(image_resize)  # 将缩放后的帧添加到对应列表
                masks_scaled[k].append(mask_resize)  # 将缩放后的遮罩添加到对应列表

        # 处理每一个去除部分
        for k in range(len(inpaint_area)):
            # 调用inpaint函数进行处理
            comps[k] = self.inpaint(frames_scaled[k], masks_scaled[k])

        # 如果存在去除部分
        if inpaint_area:
            for j in range(len(frames_hr)):
                frame = frames_hr[j]  # 取出原始帧
                # 对于模式中的每一个段落
                for k in range(len(inpaint_area)):
                    y0, y1, x0, x1 = inpaint_area[k]
                    crop_h, crop_w = y1 - y0, x1 - x0
                    # 模型输出高于裁剪时缩小用 AREA，低于裁剪时放大用 CUBIC。
                    interp = cv2.INTER_AREA if crop_h <= self.model_input_height else cv2.INTER_CUBIC
                    comp = cv2.resize(comps[k][j], (crop_w, crop_h), interpolation=interp)
                    if comp.dtype != np.uint8:
                        comp = np.clip(comp, 0, 255).astype(np.uint8)
                    # 预测是 RGB，OpenCV 帧是 BGR。这一步只换预测的通道，原帧像素不动。
                    comp = cv2.cvtColor(comp, cv2.COLOR_BGR2RGB)
                    frame[y0:y1, x0:x1] = self._paste_masked(frame[y0:y1, x0:x1], comp, mask[y0:y1, x0:x1])
                # 将最终帧添加到列表
                inpainted_frames.append(frame)
                # print(f'processing frame, {len(frames_hr) - j} left')
        else:
            inpainted_frames = frames_hr
        return inpainted_frames

    @staticmethod
    def _prefer_row_fill(frames, mask):
        """扁长遮罩且周围以横向边缘为主时，按行插值比重建模型更干净。"""
        plane = mask[:, :, 0] if mask.ndim == 3 else mask
        if plane.size == 0 or not np.any(plane):
            return False
        ys, xs = np.where(plane > 0)
        band_h = int(ys.max() - ys.min() + 1)
        band_w = int(xs.max() - xs.min() + 1)
        # 抖动会把同一行字幕的并集拉高，所以只拒绝又高又方的块。边缘方向才是真正的判断。
        if band_h > 200 or band_w < band_h * 1.2:
            return False
        step = max(1, len(frames) // 5)
        sample = frames[::step][:5]
        votes = 0
        limit = 1 if np.max(plane) <= 1 else 127
        hard = (plane > limit).astype(np.uint8) * 255
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 9))
        dilated = cv2.dilate(hard, kernel)
        ring = (dilated > 0) & (hard == 0)
        for frame in sample:
            if int(ring.sum()) < 30:
                votes += 1
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
            ax = np.abs(gx[ring])
            ay = np.abs(gy[ring])
            strong = (ax > 12) | (ay > 12)
            if int(strong.sum()) < 20:
                votes += 1
                continue
            ax = ax[strong]
            ay = ay[strong]
            horiz = float(np.mean(ay > ax * 1.3))
            vert = float(np.mean(ax > ay * 1.3))
            if horiz >= 0.62 and vert <= 0.32:
                votes += 1
        return votes >= max(1, (len(sample) + 1) // 2)

    @staticmethod
    def _row_component_mask(frames, mask):
        plane = mask[:, :, 0] if mask.ndim == 3 else mask
        limit = 1 if plane.size and np.max(plane) <= 1 else 127
        binary = (plane > limit).astype(np.uint8)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        keep = np.zeros(binary.shape, np.uint8)
        for i in range(1, count):
            x, y, w, h, area = (int(v) for v in stats[i])
            if area < 30:
                continue
            comp = np.zeros(binary.shape, np.uint8)
            comp[labels == i] = 255
            use_rows = STTNDetInpaint._prefer_row_fill(frames, comp)
            print(f"STTN {'row fill' if use_rows else 'model'} {w}x{h}")
            if use_rows:
                keep[labels == i] = 255
        return keep

    @staticmethod
    def _fill_rows_frame(frame, mask):
        """用同一行遮罩左右的像素补上字幕，横向的门线和高光会连过去。"""
        out = frame.copy()
        plane = mask[:, :, 0] if mask.ndim == 3 else mask
        limit = 1 if np.max(plane) <= 1 else 127
        hard = plane > limit
        width = hard.shape[1]
        for y in np.flatnonzero(hard.any(axis=1)):
            xs = np.flatnonzero(hard[y])
            cuts = np.flatnonzero(np.diff(xs) > 1)
            starts = np.r_[xs[0], xs[cuts + 1]]
            ends = np.r_[xs[cuts], xs[-1]] + 1
            for x0, x1 in zip(starts.tolist(), ends.tolist()):
                n = x1 - x0
                left = out[y, x0 - 1].astype(np.float32) if x0 > 0 else None
                right = out[y, x1].astype(np.float32) if x1 < width else None
                if left is None and right is None:
                    continue
                if left is None:
                    left = right
                if right is None:
                    right = left
                t = (np.arange(1, n + 1, dtype=np.float32) / (n + 1))[:, None]
                out[y, x0:x1] = (left * (1.0 - t) + right * t).astype(np.uint8)
        return out

    @staticmethod
    def _tighten_areas(areas, mask, min_pad=96):
        """把每条修复带收到字幕上下，高度尽量接近模型的 240 像素。"""
        plane = mask[:, :, 0] if mask.ndim == 3 else mask
        height = plane.shape[0]
        tight = []
        for y0, y1, x0, x1 in areas:
            rows = np.where((plane[y0:y1] > 0).any(axis=1))[0]
            if len(rows) == 0:
                continue
            top = y0 + int(rows[0])
            bot = y0 + int(rows[-1]) + 1
            pad = max(min_pad, bot - top)
            ny0 = max(0, top - pad)
            ny1 = min(height, bot + pad)
            if ny1 - ny0 < 96:
                ny0 = max(0, ny1 - 96)
                ny1 = min(height, ny0 + 96)
            tight.append((int(ny0), int(ny1), int(x0), int(x1)))
        return tight

    @staticmethod
    def _paste_masked(dst, comp, mask_area):
        """只替换遮罩内的像素，字幕上下的原图像素保持不动。"""
        plane = mask_area[:, :, 0] if mask_area.ndim == 3 else mask_area
        if plane.size == 0 or not np.any(plane):
            return dst
        limit = 1 if np.max(plane) <= 1 else 127
        hard = (plane > limit).astype(np.float32)
        # 遮罩比字形宽了约 10 像素，1.5 的羽化只会柔化接缝，不会把字带回来。
        alpha = cv2.GaussianBlur(hard, (0, 0), 1.5)[..., None]
        return np.clip(comp.astype(np.float32) * alpha + dst.astype(np.float32) * (1.0 - alpha), 0, 255).astype(np.uint8)

    @staticmethod
    def read_mask(path):
        img = cv2.imread(path, 0)
        # 转为binary mask
        ret, img = cv2.threshold(img, 127, 1, cv2.THRESH_BINARY)
        img = img[:, :, None]
        return img

    def get_ref_index(self, neighbor_ids, length):
        """
        采样整个视频的参考帧
        """
        # 初始化参考帧的索引列表
        ref_index = []
        # 在视频长度范围内根据ref_length逐步迭代
        for i in range(0, length, self.ref_length):
            # 如果当前帧不在近邻帧中
            if i not in neighbor_ids:
                # 将它添加到参考帧列表
                ref_index.append(i)
        # 返回参考帧索引列表
        return ref_index

    def inpaint(self, frames: List[np.ndarray], masks: List[np.ndarray]):
        """
        使用STTN完成空洞填充（空洞即被遮罩的区域）
        """
        frame_length = len(frames)
        # 对帧进行预处理转换为张量，并进行归一化
        feats = _to_tensors(frames).unsqueeze(0) * 2 - 1

        binary_masks = [np.expand_dims((np.array(m) > 0.5).astype(np.uint8), 2) for m in masks]
        # 将掩码转换为张量
        masks_tensor = (_to_tensors(masks).unsqueeze(0) > 0.5).float()

        # 把特征张量转移到指定的设备（CPU或GPU）
        feats, masks_tensor = feats.to(self.device), masks_tensor.to(self.device)
        # 初始化一个与视频长度相同的列表，用于存储处理完成的帧
        comp_frames = [None] * frame_length
        # 统一关闭梯度计算，用于推理阶段节省内存并加速
        with torch.no_grad():
            # 将处理好的帧通过编码器，产生特征表示
            feats = self.model.encoder((feats*(1-masks_tensor).float()).view(frame_length, 3, self.model_input_height, self.model_input_width))
            # 获取特征维度信息
            _, c, feat_h, feat_w = feats.size()
            # 调整特征形状以匹配模型的期望输入
            feats = feats.view(1, frame_length, c, feat_h, feat_w)
            # 在设定的邻居帧步幅内循环处理视频
            for f in range(0, frame_length, self.neighbor_stride):
                # 计算邻近帧的ID
                neighbor_ids = [i for i in range(max(0, f - self.neighbor_stride), min(frame_length, f + self.neighbor_stride + 1))]
                # 获取参考帧的索引
                ref_ids = self.get_ref_index(neighbor_ids, frame_length)
                # 通过模型推断特征并传递给解码器以生成完成的帧
                pred_feat = self.model.infer(
                    feats[0, neighbor_ids + ref_ids, :, :, :], masks_tensor[0, neighbor_ids + ref_ids, :, :, :])

                # 将预测的特征通过解码器生成图片，并应用激活函数tanh
                pred_img = torch.tanh(self.model.decoder(pred_feat[:len(neighbor_ids), :, :, :]))
                # 将结果张量重新缩放到0到255的范围内（图像像素值）
                pred_img = (pred_img + 1) / 2
                # 将张量移动回CPU并转为NumPy数组
                pred_img = pred_img.cpu().permute(0, 2, 3, 1).numpy() * 255
                # 遍历邻近帧
                for i in range(len(neighbor_ids)):
                    idx = neighbor_ids[i]
                    # 将预测的图片转换为无符号8位整数格式
                    img = pred_img[i].astype(np.uint8) * binary_masks[idx] + frames[idx] * (1 - binary_masks[idx])
                    if comp_frames[idx] is None:
                        comp_frames[idx] = img
                    else:
                        comp_frames[idx] = comp_frames[idx].astype(np.float32) * 0.5 + img.astype(np.float32) * 0.5
        # 返回处理完成的帧序列
        return comp_frames
