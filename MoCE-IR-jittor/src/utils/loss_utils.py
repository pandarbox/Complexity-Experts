import math

import jittor as jt
import jittor.nn as nn


class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_var is None) or (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = jt.full(input.shape, self.real_label)
                self.real_label_var = real_tensor
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = jt.full(input.shape, self.fake_label)
                self.fake_label_var = fake_tensor
            target_tensor = self.fake_label_var
        return target_tensor

    def execute(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor)


class FocalL1Loss(nn.Module):
    def __init__(self, gamma=2.0, epsilon=1e-6, alpha=0.1):
        """
        Focal L1 Loss with adjusted weighting for output values in [0, 1].
        
        Args:
            gamma (float): Focusing parameter. Larger gamma focuses more on hard examples.
            epsilon (float): Small constant to prevent weights from being zero.
            alpha (float): Scaling factor to normalize error values.
        """
        super(FocalL1Loss, self).__init__()
        self.gamma = gamma
        self.epsilon = epsilon
        self.alpha = alpha  # Scaling factor to prevent error values from being too small.

    def execute(self, pred, target):
        """
        Compute the Focal L1 Loss between the predicted and target images.
        
        Args:
            pred (jt.Var): Predicted image [b, c, h, w].
            target (jt.Var): Ground truth image [b, c, h, w].
        
        Returns:
            jt.Var: Scalar Focal L1 Loss.
        """
        # Compute the absolute error (L1 Loss) and scale it by alpha
        abs_err = jt.abs(pred - target) / self.alpha
        
        # Apply a logarithmic transformation to the error to prevent very small weights
        focal_weight = (jt.log(1 + abs_err + self.epsilon)) ** self.gamma
        
        # Compute the weighted loss
        focal_l1_loss = focal_weight * abs_err
        
        # Return the mean loss across all pixels
        return focal_l1_loss.mean()
    
    
class FFTLoss(nn.Module):
    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(FFTLoss, self).__init__()
        self.loss_weight = loss_weight
        self.reduction = reduction

    def execute(self, pred, target):
        '''
        # Jittor 使用 jt.fft.rfft2 进行 FFT 变换
        pred_fft = jt.fft.rfft2(pred)
        target_fft = jt.fft.rfft2(target)

        # 分离实部和虚部
        pred_fft = jt.stack([pred_fft.real, pred_fft.imag], dim=-1)
        target_fft = jt.stack([target_fft.real, target_fft.imag], dim=-1)
        '''
        # 使用 nn._fft2：先构造复数 (最后一维为 [real, imag])
        pred_complex = jt.stack([pred, jt.zeros_like(pred)], dim=-1)
        target_complex = jt.stack([target, jt.zeros_like(target)], dim=-1)
        pred_fft = nn._fft2(pred_complex)
        target_fft = nn._fft2(target_complex)
        # pred_fft[...,0] 为实部，[...,1] 为虚部；将两者按最后一维组织一致性
        pred_fft = jt.stack([pred_fft[..., 0], pred_fft[..., 1]], dim=-1)
        target_fft = jt.stack([target_fft[..., 0], target_fft[..., 1]], dim=-1)
        
        
        # 计算 L1 损失
        if self.reduction == 'mean':
            loss = jt.abs(pred_fft - target_fft).mean()
        elif self.reduction == 'sum':
            loss = jt.abs(pred_fft - target_fft).sum()
        else:
            loss = jt.abs(pred_fft - target_fft)

        return self.loss_weight * loss
    
    
class TemperatureScheduler:
    def __init__(self, start_temp, end_temp, total_steps):
        """
        Scheduler for Gumbel-Softmax temperature that decreases using a cosine annealing schedule.

        Args:
        - start_temp (float): Initial temperature (e.g., 5.0).
        - end_temp (float): Final temperature (e.g., 0.01).
        - total_steps (int): Total number of steps/epochs to anneal over.
        """
        self.start_temp = start_temp
        self.end_temp = end_temp
        self.total_steps = total_steps

    def get_temperature(self, step):
        """
        Get the temperature value for the current step, following a cosine annealing schedule.
        
        Args:
        - step (int): Current step or epoch.
        
        Returns:
        - temperature (float): The temperature for the Gumbel-Softmax at this step.
        """
        if step >= self.total_steps:
            return self.end_temp
        
        # Cosine annealing formula to compute the temperature
        cos_inner = math.pi * step / self.total_steps
        temp = self.start_temp + 0.5 * (self.end_temp - self.start_temp) * (1 - math.cos(cos_inner))
        
        return temp