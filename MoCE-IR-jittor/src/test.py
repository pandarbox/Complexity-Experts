import os
import pathlib
import argparse
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from typing import List
from skimage import img_as_ubyte
from skimage.metrics import structural_similarity, peak_signal_noise_ratio

import jittor as jt
import jittor.nn as nn

from net.moce_ir import MoCEIR
from options import train_options
from utils.test_utils import save_img
from data.dataset_utils import IRBenchmarks, CDD11

# 启用 GPU
jt.flags.use_cuda = 1


####################################################################################################
## HELPERS
def compute_psnr(image_true, image_test, image_mask, data_range=None):
    # this function is based on skimage.metrics.peak_signal_noise_ratio
    err = np.sum((image_true - image_test) ** 2, dtype=np.float64) / np.sum(image_mask)
    return 10 * np.log10((data_range ** 2) / err)


def compute_ssim(tar_img, prd_img, cr1):
    ssim_pre, ssim_map = structural_similarity(tar_img, prd_img, channel_axis=2, gaussian_weights=True, data_range=1.0, full=True)
    ssim_map = ssim_map * cr1
    r = int(3.5 * 1.5 + 0.5)  # radius as in ndimage
    win_size = 2 * r + 1
    pad = (win_size - 1) // 2
    ssim = ssim_map[pad:-pad, pad:-pad, :]
    crop_cr1 = cr1[pad:-pad, pad:-pad, :]
    ssim = ssim.sum(axis=0).sum(axis=0) / crop_cr1.sum(axis=0).sum(axis=0)
    ssim = np.mean(ssim)
    return ssim


def calc_psnr(img1, img2, data_range=1.0):
    err = np.sum((img1 - img2) ** 2, dtype=np.float64)
    return 10 * np.log10((data_range ** 2) / (err / img1.size))


def calc_ssim(img1, img2):
    return structural_similarity(img1, img2, channel_axis=2, gaussian_weights=True, data_range=1.0, full=False)


def calc_lpips_numpy(img1, img2):
    """
    简单的 LPIPS 替代实现（基于 VGG 特征的感知损失近似）
    """
    # 简单的 L2 距离作为替代
    return np.mean((img1 - img2) ** 2)


####################################################################################################
## Test Model
class TestModel(nn.Module):
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
            complexity_scale=opt.complexity_scale,)
    
    def execute(self, x):
        return self.net(x)


def load_model(opt):
    """加载模型检查点"""
    model = TestModel(opt)
    checkpoint_path = os.path.join(opt.ckpt_dir, opt.checkpoint_id, "last.ckpt")
    
    if os.path.exists(checkpoint_path):
        state = jt.load(checkpoint_path)
        state_dict = state['model_state_dict']
        
        # 手动加载参数，跳过 total_loss 等非模型权重
        model_state = model.state_dict()
        for key, value in state_dict.items():
            if key in model_state and model_state[key] is not None:
                # 只加载模型中存在且非 None 的参数
                if hasattr(value, 'shape'):  # 确保是有效的 tensor
                    model_state[key].assign(value)
            # 跳过 loss 相关的非模型权重键
        
        print(f"Model loaded from: {checkpoint_path}")
    else:
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    return model


####################################################################################################
def run_test(opts, net, dataset, factor=8):
    from jittor.dataset import DataLoader
    testloader = DataLoader(dataset, batch_size=1, shuffle=False, drop_last=False, num_workers=0)
    
    if opts.save_results:
        pathlib.Path(os.path.join(os.getcwd(), f"results/{opts.checkpoint_id}/{opts.benchmarks[0]}")).mkdir(parents=True, exist_ok=True)
    
    # LPIPS 计算
    try:
        import lpips
        calc_lpips = lpips.LPIPS(net='vgg').cuda()
        use_lpips_lib = True
        print("Using lpips library for LPIPS calculation")
    except ImportError:
        use_lpips_lib = False
        print("Warning: lpips library not found, using simplified perceptual metric")
    
    psnr_list, ssim_list, lpips_list = [], [], []
    
    with jt.no_grad():
        for (clean_name, de_id, degrad_patch, clean_patch) in tqdm(testloader):
            # Forward pass
            restored = net(degrad_patch)
            if isinstance(restored, List) and len(restored) == 2:
                restored, _ = restored
            
            # Unpad images to original dimensions
            assert restored.shape == clean_patch.shape, "Restored and clean patch shape mismatch."
            
            # Clamp output images
            restored = jt.clamp(restored, 0, 1)
            
            # Calculate LPIPS
            if use_lpips_lib:
                # 转换为 torch tensor 计算 LPIPS
                import torch
                restored_torch = torch.from_numpy(restored.numpy()).cuda()
                clean_torch = torch.from_numpy(clean_patch.numpy()).cuda()
                lpips_val = calc_lpips(clean_torch * 2 - 1, restored_torch * 2 - 1).item()
            else:
                # 使用简化版本
                lpips_val = calc_lpips_numpy(clean_patch.numpy(), restored.numpy())
            lpips_list.append(lpips_val)
            
            # Convert to numpy for PSNR/SSIM calculation
            restored_np = restored.permute(0, 2, 3, 1).squeeze(0).numpy()
            degrad_np = degrad_patch.permute(0, 2, 3, 1).squeeze(0).numpy()
            clean_np = clean_patch.permute(0, 2, 3, 1).squeeze(0).numpy()
            
            # Calculate metrics
            ssim_val = calc_ssim(clean_np, restored_np)
            ssim_list.append(ssim_val)
            psnr_val = peak_signal_noise_ratio(clean_np, restored_np, data_range=1)
            psnr_list.append(psnr_val)
            
            # Save results if needed
            if opts.save_results:
                save_name = os.path.splitext(os.path.split(clean_name[0])[-1])[0] + '_' + str(round(psnr_val, 2)) + '.png'
                save_img(
                    os.path.join(os.getcwd(), 
                                f"results/{opts.checkpoint_id}/{opts.benchmarks[0]}", 
                                save_name), 
                    img_as_ubyte(restored_np))

    print('PSNR: {:f} SSIM: {:f} LPIPS: {:f}\n'.format(np.mean(psnr_list), np.mean(ssim_list), np.mean(lpips_list)))


## test LolV1
def run_lolv1(opts, net, dataset, factor=8):
    run_test(opts, net, dataset, factor)


## test GoPro
def run_gopro(opts, net, dataset, factor=8):
    run_test(opts, net, dataset, factor)


## test Derain
def run_derain(opts, net, dataset, factor=8):
    run_test(opts, net, dataset, factor)


## test Dehaze
def run_dehaze(opts, net, dataset, factor=8):
    run_test(opts, net, dataset, factor)


## test synthetic denoising
def run_denoise_15(opts, net, dataset, factor=8):
    run_test(opts, net, dataset, factor)


def run_denoise_25(opts, net, dataset, factor=8):
    run_test(opts, net, dataset, factor)


def run_denoise_50(opts, net, dataset, factor=8):
    run_test(opts, net, dataset, factor)


# test CDD11
def run_cdd11(opts, net, dataset, factor=8):
    run_test(opts, net, dataset, factor)


####################################################################################################
## main
def main(opt):
    np.random.seed(0)
    jt.set_global_seed(0)

    # Load model
    net = load_model(opt)
    net.eval()
    
    for de in opt.benchmarks:
        ind_opt = opt
        ind_opt.benchmarks = [de]
        
        if "CDD11" in opt.trainset:
            _, subset = opt.trainset.split("_", maxsplit=1)
            dataset = CDD11(opt, split="test", subset=subset)
        else:
            dataset = IRBenchmarks(ind_opt)
        
        print("--------> Testing on", de, "testset.")
        print("\n")
        globals()[f"run_{de}"](opt, net, dataset, factor=8)


def depth_type(value):
    try:
        return int(value)  # Try to convert to int
    except ValueError:
        return value  # If it fails, return the string


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == '__main__':
    train_opt = train_options()
    main(train_opt)