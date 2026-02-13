#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MoCE-IR (Jittor版本) 图像复原脚本
支持三种任务:
1. 去雨 (derain): 加载退化图像，复原并保存
2. 去雾 (dehaze): 加载退化图像，复原并保存  
3. 去噪 (denoise): 加载原图，添加噪声，保存噪声图像，复原并保存

使用方法:
    # 去雨任务
    python src/restoration.py --task derain \
        --input datasets/deraining/Rain100L/rainy/rain-001.png \
        --gt datasets/deraining/Rain100L/gt/norain-001.png \
        --checkpoint checkpoints/2026_02_06_02_47_48/last.ckpt
    
    # 去雾任务
    python src/restoration.py --task dehaze \
        --input datasets/dehazing/SOTS/outdoor/hazy/0019_0.8_0.16.jpg \
        --gt datasets/dehazing/SOTS/outdoor/gt/0019.png \
        --checkpoint checkpoints/2026_02_06_02_47_48/last.ckpt
    
    # 去噪任务
    python src/restoration.py --task denoise \
        --input datasets/denoising/cBSD68/original_png/101085.png \
        --noise_level 25 \
        --checkpoint checkpoints/2026_02_06_02_47_48/last.ckpt
"""


import os
import sys
import argparse
import numpy as np
import cv2
from datetime import datetime
from types import SimpleNamespace
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

import jittor as jt
from jittor import nn

# 启用 GPU
jt.flags.use_cuda = 1

# 添加src目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from net.moce_ir import MoCEIR


################################################################################
# 模型定义
################################################################################
class TestModel(nn.Module):
    """用于推理的Jittor模型封装"""
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
    
    def execute(self, x):
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
    """向图像添加高斯噪声"""
    noise = np.random.randn(*clean_img.shape) * sigma
    noisy_img = np.clip(clean_img + noise, 0, 255).astype(np.uint8)
    return noisy_img


def save_img(filepath: str, img: np.ndarray):
    """保存图像 (RGB格式输入)"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    cv2.imwrite(filepath, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    print(f"[保存] {filepath}")


def load_img(filepath: str) -> np.ndarray:
    """加载图像，返回RGB格式"""
    img = cv2.imread(filepath)
    if img is None:
        raise FileNotFoundError(f"无法读取图像: {filepath}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def calc_psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    """计算PSNR (输入范围 [0, 255])"""
    return peak_signal_noise_ratio(img1, img2, data_range=255)


def calc_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    """计算SSIM (输入范围 [0, 255])"""
    return structural_similarity(img1, img2, channel_axis=2, data_range=255)


################################################################################
# 模型加载
################################################################################
def get_default_opt(model_type: str = "MoCE_IR_S"):
    """获取默认模型配置"""
    if model_type == "MoCE_IR_S":
        cfg = {
            "dim": 32, "num_blocks": [4, 6, 6, 8], "num_dec_blocks": [2, 4, 4],
            "latent_dim": 2, "num_exp_blocks": 4, "num_refinement_blocks": 4,
            "heads": [1, 2, 4, 8], "stage_depth": [1, 1, 1], "with_complexity": False,
            "complexity_scale": "max", "rank_type": "spread", "depth_type": "constant",
            "topk": 1
        }
    else:  # MoCE_IR
        cfg = {
            "dim": 48, "num_blocks": [4, 6, 6, 8], "num_dec_blocks": [2, 4, 4],
            "latent_dim": 2, "num_exp_blocks": 4, "num_refinement_blocks": 4,
            "heads": [1, 2, 4, 8], "stage_depth": [1, 1, 1], "with_complexity": False,
            "complexity_scale": "max", "rank_type": "spread", "depth_type": "constant",
            "topk": 1
        }
    return SimpleNamespace(**cfg)


def load_model(checkpoint_path: str, model_type: str = "MoCE_IR_S"):
    """加载训练好的模型"""
    opt = get_default_opt(model_type)
    
    print(f"[加载模型] {checkpoint_path}")
    model = TestModel(opt)
    
    if os.path.exists(checkpoint_path):
        state = jt.load(checkpoint_path)
        state_dict = state['model_state_dict']
        
        # 加载参数
        model_state = model.state_dict()
        for key, value in state_dict.items():
            if key in model_state and model_state[key] is not None:
                if hasattr(value, 'shape'):
                    model_state[key].assign(value)
        
        print("[模型加载完成]")
    else:
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    # 设置为评估模式
    model.eval()
    
    return model


################################################################################
# 复原函数
################################################################################
def restore_image(model, degraded_img: np.ndarray) -> np.ndarray:
    """使用模型复原图像"""
    # 裁剪到16的倍数
    degraded_img = crop_img(degraded_img, base=16)
    
    # 转换为tensor (H,W,C) -> (1,C,H,W), 归一化到[0,1]
    img_float = degraded_img.astype(np.float32) / 255.0
    input_tensor = jt.array(img_float).permute(2, 0, 1).unsqueeze(0)
    
    with jt.no_grad():
        restored = model(input_tensor)
        
        # 处理模型可能返回tuple的情况
        if isinstance(restored, (list, tuple)):
            restored = restored[0]
        
        # 后处理
        restored = jt.clamp(restored, 0, 1)
        restored = restored.permute(0, 2, 3, 1).squeeze(0).numpy()
        restored = (restored * 255).astype(np.uint8)
    
    return restored


################################################################################
# 任务处理函数
################################################################################
def process_derain(model, input_path: str, output_dir: str, gt_path: str = None):
    """去雨任务处理"""
    print(f"\n{'='*60}")
    print(f"[去雨任务] 处理: {input_path}")
    print(f"{'='*60}")
    
    rainy_img = load_img(input_path)
    print(f"[输入图像] 尺寸: {rainy_img.shape}")
    
    restored_img = restore_image(model, rainy_img)
    print(f"[复原图像] 尺寸: {restored_img.shape}")
    
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


def process_dehaze(model, input_path: str, output_dir: str, gt_path: str = None):
    """去雾任务处理"""
    print(f"\n{'='*60}")
    print(f"[去雾任务] 处理: {input_path}")
    print(f"{'='*60}")
    
    hazy_img = load_img(input_path)
    print(f"[输入图像] 尺寸: {hazy_img.shape}")
    
    restored_img = restore_image(model, hazy_img)
    print(f"[复原图像] 尺寸: {restored_img.shape}")
    
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


def process_denoise(model, input_path: str, output_dir: str, noise_level: int):
    """去噪任务处理"""
    print(f"\n{'='*60}")
    print(f"[去噪任务] 处理: {input_path}")
    print(f"[噪声级别] sigma = {noise_level}")
    print(f"{'='*60}")
    
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
    restored_img = restore_image(model, noisy_img)
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
    parser = argparse.ArgumentParser(description='MoCE-IR (Jittor) 图像复原脚本')
    
    parser.add_argument('--task', type=str, required=True, 
                        choices=['derain', 'dehaze', 'denoise'],
                        help='任务类型: derain(去雨), dehaze(去雾), denoise(去噪)')
    parser.add_argument('--input', type=str, required=True,
                        help='输入图像路径')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='输出目录 (默认: results/<task>/<timestamp>)')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='模型checkpoint路径')
    parser.add_argument('--model', type=str, default='MoCE_IR_S',
                        choices=['MoCE_IR', 'MoCE_IR_S'],
                        help='模型类型')
    parser.add_argument('--noise_level', type=int, default=25,
                        help='噪声级别 (仅用于denoise任务)')
    parser.add_argument('--gt', type=str, default=None,
                        help='GT图像路径 (可选，用于derain/dehaze计算PSNR/SSIM)')
    
    args = parser.parse_args()
    
    # 检查输入文件
    if not os.path.exists(args.input):
        parser.error(f"输入文件不存在: {args.input}")
    
    if not os.path.exists(args.checkpoint):
        parser.error(f"模型文件不存在: {args.checkpoint}")
    
    # 设置输出目录
    if args.output_dir is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.output_dir = f'results/{args.task}/{timestamp}'
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载模型
    model = load_model(args.checkpoint, args.model)
    
    # 处理任务
    if args.task == 'derain':
        process_derain(model, args.input, args.output_dir, args.gt)
    elif args.task == 'dehaze':
        process_dehaze(model, args.input, args.output_dir, args.gt)
    elif args.task == 'denoise':
        process_denoise(model, args.input, args.output_dir, args.noise_level)
    
    print(f"\n[完成] 结果保存在: {args.output_dir}")


if __name__ == '__main__':
    main()
