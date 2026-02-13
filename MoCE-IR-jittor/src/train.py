from typing import List

import os
import pathlib
import numpy as np
import json
import logging
import random

from tqdm import tqdm
from datetime import datetime

import jittor as jt
import jittor.nn as nn
from jittor import optim
from jittor.dataset import DataLoader

from tensorboardX import SummaryWriter

from net.moce_ir import MoCEIR

from options import train_options
from utils.schedulers import LinearWarmupCosineAnnealingLR
from data.dataset_utils import AIOTrainDataset, CDD11
from utils.loss_utils import FFTLoss


# ============ 性能优化设置 ============
# 启用 GPU
jt.flags.use_cuda = 1
# 启用 cuDNN 加速
jt.cudnn.set_max_workspace_ratio(0.1)
# 设置 Jittor 编译优化
jt.flags.lazy_execution = 1


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    jt.set_global_seed(seed)

def setup_logger(log_dir, name="train"):
    """设置日志记录器"""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # 文件处理器
    fh = logging.FileHandler(os.path.join(log_dir, "train.log"))
    fh.setLevel(logging.INFO)
    
    # 控制台处理器
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    # 格式化器
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger


class TrainModel(nn.Module):
    
    def __init__(self, opt):
        super().__init__()
        
        self.opt = opt
        self.balance_loss_weight = opt.balance_loss_weight

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
        
        if opt.loss_type == "fft":
            self.loss_fn = nn.L1Loss()
            self.aux_fn = FFTLoss(loss_weight=self.opt.fft_loss_weight)
        else:
            self.loss_fn = nn.L1Loss()
    
    def execute(self, x, de_id=None):
        return self.net(x, de_id)
    
    def compute_loss(self, degrad_patch, clean_patch, de_id):
        restored = self.net(degrad_patch, de_id)
        balance_loss = self.net.total_loss

        if self.opt.loss_type == "fft":
            loss = self.loss_fn(restored, clean_patch)
            aux_loss = self.aux_fn(restored, clean_patch)
            loss += aux_loss
        else:
            loss = self.loss_fn(restored, clean_patch)
            
        loss += self.balance_loss_weight * balance_loss
        return loss, balance_loss


def save_checkpoint(model, optimizer, epoch, global_step, checkpoint_path, filename="last.ckpt"):
    """保存检查点"""
    filepath = os.path.join(checkpoint_path, filename)
    state = {
        'epoch': epoch,
        'global_step': global_step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    jt.save(state, filepath)
    return filepath


def load_checkpoint(model, optimizer, checkpoint_path):
    """加载检查点"""
    if os.path.exists(checkpoint_path):
        state = jt.load(checkpoint_path)
        model.load_state_dict(state['model_state_dict'])
        if optimizer is not None and 'optimizer_state_dict' in state:
            optimizer.load_state_dict(state['optimizer_state_dict'])
        start_epoch = state.get('epoch', 0) + 1
        global_step = state.get('global_step', 0)
        return start_epoch, global_step
    return 0, 0


def main(opt):
    print("Options")
    print(opt)
    time_stamp = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
        
    log_dir = os.path.join("logs/", time_stamp)
    pathlib.Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    # 设置日志记录器
    logger = setup_logger(log_dir)
    
    # 设置 TensorBoard 日志记录器
    tb_log_dir = os.path.join(log_dir, "tensorboard")
    pathlib.Path(tb_log_dir).mkdir(parents=True, exist_ok=True)
    tb_writer = SummaryWriter(log_dir=tb_log_dir)
    logger.info(f"TensorBoard logs will be saved to: {tb_log_dir}")
    
    # 保存训练配置
    config_path = os.path.join(log_dir, "config.json")
    with open(config_path, 'w') as f:
        json.dump(vars(opt), f, indent=4, default=str)
    logger.info(f"Config saved to: {config_path}")
    
    # 将配置写入 TensorBoard
    tb_writer.add_text('Config', json.dumps(vars(opt), indent=4, default=str), 0)

    # Create model
    model = TrainModel(opt)
    
    # 加载预训练模型（fine-tune）
    if opt.fine_tune_from:
        fine_tune_path = os.path.join(opt.ckpt_dir, opt.fine_tune_from, "last.ckpt")
        if os.path.exists(fine_tune_path):
            state = jt.load(fine_tune_path)
            model.load_state_dict(state['model_state_dict'])
            logger.info(f"Loaded fine-tune model from: {fine_tune_path}")

    logger.info(f"Model created:\n{model}")
    
    # 计算模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Total parameters: {total_params / 1e6:.2f}M")
    
    # 记录模型参数量到 TensorBoard
    tb_writer.add_text('Model/Parameters', f'Total: {total_params / 1e6:.2f}M', 0)
    
    checkpoint_path = os.path.join(opt.ckpt_dir, time_stamp)
    pathlib.Path(checkpoint_path).mkdir(parents=True, exist_ok=True)
    logger.info(f"Checkpoints will be saved to: {checkpoint_path}")
    
    # Create datasets and dataloaders
    if "CDD11" in opt.trainset:
        _, subset = opt.trainset.split("_")
        trainset = CDD11(opt, split="train", subset=subset)
    else:
        trainset = AIOTrainDataset(opt)
    
    trainloader = DataLoader(
        trainset, 
        batch_size=opt.batch_size, 
        shuffle=True, 
        drop_last=True, 
        num_workers=opt.num_workers
    )
    
    num_batches = len(trainset) // opt.batch_size
    logger.info(f"Dataset loaded: {len(trainset)} samples, {num_batches} batches per epoch")
    
    # Create optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=2e-4)
    
    if opt.fine_tune_from:
        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer=optimizer, warmup_epochs=1, max_epochs=opt.epochs
        )
    else:
        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer=optimizer, warmup_epochs=15, max_epochs=150
        )
    
    # Optionally resume from a checkpoint
    start_epoch = 0
    global_step = 0
    if opt.resume_from:
        resume_path = os.path.join(opt.ckpt_dir, opt.resume_from, "last.ckpt")
        start_epoch, global_step = load_checkpoint(model, optimizer, resume_path)
        logger.info(f"Resumed from checkpoint: {resume_path}, epoch {start_epoch}, step {global_step}")

    # Training loop
    accum_steps = opt.accum_grad if hasattr(opt, 'accum_grad') else 1
    logger.info(f"Starting training from epoch {start_epoch + 1} to {opt.epochs}")
    logger.info(f"Gradient accumulation steps: {accum_steps}")
    
    
    best_loss = float('inf')
    
    for epoch in range(start_epoch, opt.epochs):

        model.train()
        epoch_loss = 0.0
        epoch_balance_loss = 0.0
        
        pbar = tqdm(enumerate(trainloader), total=num_batches, desc=f"Epoch {epoch+1}/{opt.epochs}")
        
        for batch_idx, batch in pbar:
            clean_name, de_id, degrad_patch, clean_patch = batch

            # 计算损失
            loss, balance_loss = model.compute_loss(degrad_patch, clean_patch, de_id)
            
            # 梯度累积
            loss = loss / accum_steps
            
            # 反向传播
            optimizer.step(loss)
            
            global_step += 1
            
            # 获取损失值
            loss_val = loss.item() * accum_steps
            balance_val = balance_loss.item() if hasattr(balance_loss, 'item') else float(balance_loss)
            current_lr = optimizer.param_groups[0].get('lr', optimizer.lr)
            
            # 更新进度条
            pbar.set_postfix({
                'Loss': f'{loss_val:.4f}',
                'Balance': f'{balance_val:.4f}',
                'LR': f'{current_lr:.6f}'
            })
            
            # 累积 epoch 统计
            epoch_loss += loss_val
            epoch_balance_loss += balance_val
            
            # 记录到 TensorBoard
            tb_writer.add_scalar('Train_Loss', loss_val, global_step)
            tb_writer.add_scalar('Balance', balance_val, global_step)
            tb_writer.add_scalar('LR Schedule', current_lr, global_step)
        
        # 更新学习率
        scheduler.step()
        
        # Epoch 结束时同步一次获取准确统计
        jt.sync_all()
        
        # 计算 epoch 统计
        avg_loss = epoch_loss / num_batches
        avg_balance = epoch_balance_loss / num_batches
        current_lr = optimizer.param_groups[0].get('lr', optimizer.lr)
        
        # 记录 epoch 级别的 TensorBoard 日志
        tb_writer.add_scalar('Epoch/Avg_Loss', avg_loss, epoch + 1)
        tb_writer.add_scalar('Epoch/Avg_Balance_Loss', avg_balance, epoch + 1)
        tb_writer.add_scalar('Epoch/LR', current_lr, epoch + 1)
        
        logger.info(f"Epoch {epoch+1}/{opt.epochs} - "
                   f"Avg Loss: {avg_loss:.4f}, "
                   f"Avg Balance: {avg_balance:.4f}, "
                   f"LR: {current_lr:.6f}, "
                   f"Global Step: {global_step}")
                  
        
        # 保存 last.ckpt
        filepath = save_checkpoint(model, optimizer, epoch, global_step, checkpoint_path, "last.ckpt")
        logger.info(f"Checkpoint saved: {filepath}")
        
        # 每5个epoch保存一次
        if (epoch + 1) % 2 == 0:
            filepath = save_checkpoint(model, optimizer, epoch, global_step, checkpoint_path, 
                                       f"epoch={epoch+1}-step={global_step}.ckpt")
            logger.info(f"Periodic checkpoint saved: {filepath}")
        
        # 保存最佳模型
        if avg_loss < best_loss:
            best_loss = avg_loss
            filepath = save_checkpoint(model, optimizer, epoch, global_step, checkpoint_path, "best.ckpt")
            logger.info(f"Best model saved: {filepath} (loss: {best_loss:.4f})")
            tb_writer.add_scalar('Epoch/Best_Loss', best_loss, epoch + 1)
        
            
    # 关闭 TensorBoard writer
    tb_writer.close()
    
    logger.info("Training completed!")
    logger.info(f"Best loss: {best_loss:.4f}")
    logger.info(f"Total steps: {global_step}")
    logger.info(f"Checkpoints saved to: {checkpoint_path}")
    logger.info(f"Logs saved to: {log_dir}")
    logger.info(f"TensorBoard logs saved to: {tb_log_dir}")
    logger.info(f"Run 'tensorboard --logdir={tb_log_dir}' to view logs")


if __name__ == '__main__':
    set_seed(0)
    train_opt = train_options()
    main(train_opt)

