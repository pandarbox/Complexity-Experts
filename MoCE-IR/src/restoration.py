#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MoCE-IR 图像复原脚本
支持三种任务:
1. 去雨 (derain): 加载退化图像，复原并保存
2. 去雾 (dehaze): 加载退化图像，复原并保存  
3. 去噪 (denoise): 加载原图，添加噪声，保存噪声图像，复原并保存

使用方法:
    # 去雨任务 (带GT计算PSNR/SSIM)
    python src/restoration.py --task derain \
        --input datasets/deraining/Rain100L/rainy/rain-001.png \
        --gt datasets/deraining/Rain100L/gt/norain-001.png \
        --checkpoint checkpoints/2025_12_18_21_51_09/last.ckpt
    
    # 去雾任务 (带GT计算PSNR/SSIM)
    python src/restoration.py --task dehaze \
        --input datasets/dehazing/SOTS/outdoor/hazy/0019_0.8_0.16.jpg  \
        --gt datasets/dehazing/SOTS/outdoor/gt/0019.png \
        --checkpoint checkpoints/2025_12_18_21_51_09/last.ckpt
    
    # 去噪任务 (自动计算PSNR/SSIM，因为原图即为GT)
    python src/restoration.py --task denoise \
        --input datasets/denoising/cBSD68/original_png/101085.png \
        --noise_level 25 \
        --checkpoint checkpoints/2025_12_18_21_51_09/last.ckpt
"""

import os
import sys
import argparse
import numpy as np
from PIL import Image
from datetime import datetime
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

import torch
import torch.nn as nn
import lightning.pytorch as pl
from torchvision.transforms import ToTensor
from skimage import img_as_ubyte

# 添加src目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from net.moce_ir import MoCEIR


################################################################################
# 模型定义
################################################################################
class PLTestModel(pl.LightningModule):
    """用于推理的PyTorch Lightning模型封装"""
    def __init__(self, opt):
        super().__init__()
        
        self.net = MoCEIR(
            dim=opt.dim, 
            num_blocks=opt.num_blocks, 
            num_dec_blocks=opt.num_dec_blocks, 
            levels=len(opt.num_blocks),
            heads=opt.heads, 
            num_refinement_blocks=opt.num_refinement_blocks, 
            topk=opt.topk, 
            num_experts=opt.num_exp_blocks,
            rank=opt.latent_dim,
            with_complexity=opt.with_complexity, 
            depth_type=opt.depth_type, 
            stage_depth=opt.stage_depth, 
            rank_type=opt.rank_type, 
            complexity_scale=opt.complexity_scale,
        )
    
    def forward(self, x):
        return self.net(x)


################################################################################
# 工具函数
################################################################################
def crop_img(image, base=16):
    """裁剪图像使其尺寸为base的倍数"""
    h = image.shape[0]
    w = image.shape[1]
    crop_h = h % base
    crop_w = w % base
    return image[crop_h // 2:h - crop_h + crop_h // 2, crop_w // 2:w - crop_w + crop_w // 2, :]


def add_gaussian_noise(clean_img: np.ndarray, sigma: int) -> np.ndarray:
    """
    向图像添加高斯噪声
    
    Args:
        clean_img: 干净图像 (H, W, C), 值范围 [0, 255]
        sigma: 噪声标准差
    
    Returns:
        噪声图像 (H, W, C), 值范围 [0, 255]
    """
    noise = np.random.randn(*clean_img.shape) * sigma
    noisy_img = np.clip(clean_img + noise, 0, 255).astype(np.uint8)
    return noisy_img


def save_img(filepath: str, img: np.ndarray):
    """
    保存图像
    
    Args:
        filepath: 保存路径
        img: 图像数组, 范围 [0, 255], uint8
    """
    import cv2
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    cv2.imwrite(filepath, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    print(f"[保存] {filepath}")


def load_img(filepath: str) -> np.ndarray:
    """
    加载图像
    
    Args:
        filepath: 图像路径
        
    Returns:
        图像数组 (H, W, C), RGB格式, 值范围 [0, 255]
    """
    img = Image.open(filepath).convert('RGB')
    return np.array(img)


def calc_psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    """计算PSNR (输入范围 [0, 255])"""
    return peak_signal_noise_ratio(img1, img2, data_range=255)


def calc_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    """计算SSIM (输入范围 [0, 255])"""
    return structural_similarity(img1, img2, channel_axis=2, data_range=255)


################################################################################
# 模型加载
################################################################################
def get_default_opt():
    """获取默认模型配置"""
    class Options:
        # 模型架构参数 (MoCE_IR_S配置)
        dim = 32
        num_blocks = [4, 6, 6, 8]
        num_dec_blocks = [2, 4, 4]
        latent_dim = 2
        num_exp_blocks = 4
        num_refinement_blocks = 4
        heads = [1, 2, 4, 8]
        stage_depth = [1, 1, 1]
        with_complexity = True
        complexity_scale = "max"
        rank_type = "spread"
        depth_type = "constant"
        topk = 1
    
    return Options()


def load_model(checkpoint_path: str, device: str = 'cuda'):
    """
    加载训练好的模型
    
    Args:
        checkpoint_path: 模型checkpoint路径
        device: 设备 ('cuda' 或 'cpu')
    
    Returns:
        加载好的模型
    """
    opt = get_default_opt()
    
    print(f"[加载模型] {checkpoint_path}")
    model = PLTestModel.load_from_checkpoint(checkpoint_path, opt=opt, map_location=device)
    model = model.to(device)
    model.eval()
    print("[模型加载完成]")
    
    return model


################################################################################
# 复原函数
################################################################################
def restore_image(model, degraded_img: np.ndarray, device: str = 'cuda') -> np.ndarray:
    """
    使用模型复原图像
    
    Args:
        model: 复原模型
        degraded_img: 退化图像 (H, W, C), RGB格式, 值范围 [0, 255]
        device: 设备
    
    Returns:
        复原图像 (H, W, C), RGB格式, 值范围 [0, 255]
    """
    to_tensor = ToTensor()
    
    # 裁剪到16的倍数
    degraded_img = crop_img(degraded_img, base=16)
    
    # 转换为tensor
    input_tensor = to_tensor(degraded_img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        restored = model(input_tensor)
        
        # 处理模型可能返回tuple的情况
        if isinstance(restored, (list, tuple)) and len(restored) == 2:
            restored, _ = restored
        
        # 后处理
        restored = torch.clamp(restored, 0, 1)
        restored = restored.cpu().detach().permute(0, 2, 3, 1).squeeze(0).numpy()
        restored = img_as_ubyte(restored)
    
    return restored


################################################################################
# 任务处理函数
################################################################################
def process_derain(model, input_path: str, output_dir: str, device: str = 'cuda', gt_path: str = None):
    """
    去雨任务处理
    
    Args:
        model: 复原模型
        input_path: 输入雨图路径
        output_dir: 输出目录
        device: 设备
        gt_path: GT图像路径 (可选，用于计算PSNR/SSIM)
    """
    print(f"\n{'='*60}")
    print(f"[去雨任务] 处理: {input_path}")
    print(f"{'='*60}")
    
    # 加载退化图像
    rainy_img = load_img(input_path)
    print(f"[输入图像] 尺寸: {rainy_img.shape}")
    
    # 复原
    restored_img = restore_image(model, rainy_img, device)
    print(f"[复原图像] 尺寸: {restored_img.shape}")
    
    # 保存结果
    basename = os.path.splitext(os.path.basename(input_path))[0]
    restored_path = os.path.join(output_dir, f"{basename}_restored.png")
    save_img(restored_path, restored_img)
    
    # 计算指标 (如果提供了GT)
    if gt_path and os.path.exists(gt_path):
        gt_img = load_img(gt_path)
        gt_img = crop_img(gt_img, base=16)
        psnr = calc_psnr(gt_img, restored_img)
        ssim = calc_ssim(gt_img, restored_img)
        print(f"\n[指标] PSNR: {psnr:.2f} dB, SSIM: {ssim:.4f}")
    
    return restored_img


def process_dehaze(model, input_path: str, output_dir: str, device: str = 'cuda', gt_path: str = None):
    """
    去雾任务处理
    
    Args:
        model: 复原模型
        input_path: 输入雾图路径
        output_dir: 输出目录
        device: 设备
        gt_path: GT图像路径 (可选，用于计算PSNR/SSIM)
    """
    print(f"\n{'='*60}")
    print(f"[去雾任务] 处理: {input_path}")
    print(f"{'='*60}")
    
    # 加载退化图像
    hazy_img = load_img(input_path)
    print(f"[输入图像] 尺寸: {hazy_img.shape}")
    
    # 复原
    restored_img = restore_image(model, hazy_img, device)
    print(f"[复原图像] 尺寸: {restored_img.shape}")
    
    # 保存结果
    basename = os.path.splitext(os.path.basename(input_path))[0]
    restored_path = os.path.join(output_dir, f"{basename}_restored.png")
    save_img(restored_path, restored_img)
    
    # 计算指标 (如果提供了GT)
    if gt_path and os.path.exists(gt_path):
        gt_img = load_img(gt_path)
        gt_img = crop_img(gt_img, base=16)
        psnr = calc_psnr(gt_img, restored_img)
        ssim = calc_ssim(gt_img, restored_img)
        print(f"\n[指标] PSNR: {psnr:.2f} dB, SSIM: {ssim:.4f}")
    
    return restored_img


def process_denoise(model, input_path: str, output_dir: str, noise_level: int, device: str = 'cuda'):
    """
    去噪任务处理
    
    Args:
        model: 复原模型
        input_path: 输入原图路径
        output_dir: 输出目录
        noise_level: 噪声级别 (标准差)
        device: 设备
    """
    print(f"\n{'='*60}")
    print(f"[去噪任务] 处理: {input_path}")
    print(f"[噪声级别] sigma = {noise_level}")
    print(f"{'='*60}")
    
    # 加载原图
    clean_img = load_img(input_path)
    clean_img = crop_img(clean_img, base=16)
    print(f"[原图] 尺寸: {clean_img.shape}")
    
    # 添加噪声
    noisy_img = add_gaussian_noise(clean_img, noise_level)
    print(f"[噪声图像] 已添加高斯噪声 (sigma={noise_level})")
    
    # 保存噪声图像
    basename = os.path.splitext(os.path.basename(input_path))[0]
    noisy_path = os.path.join(output_dir, f"{basename}_noisy_sigma{noise_level}.png")
    save_img(noisy_path, noisy_img)
    
    # 复原
    restored_img = restore_image(model, noisy_img, device)
    print(f"[复原图像] 尺寸: {restored_img.shape}")
    
    # 保存复原图像
    restored_path = os.path.join(output_dir, f"{basename}_restored_sigma{noise_level}.png")
    save_img(restored_path, restored_img)
    
    # 保存原图用于对比
    clean_path = os.path.join(output_dir, f"{basename}_clean.png")
    save_img(clean_path, clean_img)
    
    # 计算指标
    psnr = calc_psnr(clean_img, restored_img)
    ssim = calc_ssim(clean_img, restored_img)
    print(f"\n[指标] PSNR: {psnr:.2f} dB, SSIM: {ssim:.4f}")
    
    return clean_img, noisy_img, restored_img


################################################################################
# 主程序
################################################################################
def main():
    parser = argparse.ArgumentParser(description='MoCE-IR 图像复原脚本')
    
    # 基本参数
    parser.add_argument('--task', type=str, required=True, 
                        choices=['derain', 'dehaze', 'denoise'],
                        help='任务类型: derain(去雨), dehaze(去雾), denoise(去噪)')
    parser.add_argument('--input', type=str, required=True,
                        help='输入图像路径')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='输出目录 (默认: results/<task>/<timestamp>)')
    
    # 模型参数
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='模型checkpoint路径 (必须指定)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='运行设备 (cuda/cpu)')
    
    # 去噪参数
    parser.add_argument('--noise_level', type=int, default=25,
                        help='噪声级别 (仅用于denoise任务): 15, 25, 50等')
    
    # GT参数
    parser.add_argument('--gt', type=str, default=None,
                        help='GT图像路径 (可选，用于derain/dehaze计算PSNR/SSIM)')
    
    args = parser.parse_args()
    
    # 检查输入文件是否存在
    if not os.path.exists(args.input):
        parser.error(f"输入文件不存在: {args.input}")
    
    # 检查checkpoint是否存在
    if not os.path.exists(args.checkpoint):
        parser.error(f"模型文件不存在: {args.checkpoint}")
    
    # 设置输出目录
    if args.output_dir is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.output_dir = f'results/{args.task}/{timestamp}'
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 检查设备
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("[警告] CUDA不可用，使用CPU")
        args.device = 'cpu'
    
    # 加载模型
    model = load_model(args.checkpoint, args.device)
    
    # 处理任务
    if args.task == 'derain':
        process_derain(model, args.input, args.output_dir, args.device, args.gt)
    elif args.task == 'dehaze':
        process_dehaze(model, args.input, args.output_dir, args.device, args.gt)
    elif args.task == 'denoise':
        process_denoise(model, args.input, args.output_dir, args.noise_level, args.device)
    
    print(f"\n[完成] 结果保存在: {args.output_dir}")


if __name__ == '__main__':
    main()
