from collections import OrderedDict
from typing import Optional, List, Tuple

import jittor as jt
from jittor import nn
import jittor.nn as F

import math
import random
import numbers
import numpy as np

from einops import rearrange
'''
from einops.layers.torch import Rearrange
from torch.distributions.normal import Normal
from fvcore.nn import FlopCountAnalysis, flop_count_table
'''

def normal_cdf(x):
    return 0.5*(1.0 + jt.erf(x/math.sqrt(2.0)))


##########################################################################
## Helper functions
def zero_module(module):
    """
    将模块的所有参数初始化为零并返回。
    """
    for p in module.parameters():
        p.assign(jt.zeros(p.shape))
    return module

class MySequential(nn.Sequential):
    """
    自定义 Sequential 类，支持传递两个输入参数。
    """
    def execute(self, x1, x2):
        # Iterate through all layers in sequential order
        for layer in self:
            # Check if the layer takes two inputs (i.e., custom layers)
            if isinstance(layer, nn.Module):
                # Pass both inputs to the layer
                x1 = layer(x1, x2)
            else:
                # For non-module layers, pass the two inputs directly
                x1 = layer(x1, x2)
        return x1

def softmax_with_temperature(logits, temperature=1.0):
    """
    Apply softmax with temperature to the logits.
    
    Args:
    - logits (torch.Tensor): The input logits.
    - temperature (float): The temperature factor.
    
    Returns:
    - torch.Tensor: The softmax output with temperature.
    """
    # Scale the logits by the temperature
    scaled_logits = logits / temperature
    
    # Apply softmax
    return F.softmax(scaled_logits, dim=-1)



class SparseDispatcher(object):
    """
    稀疏分发器：用于将输入样本分发给不同的专家，并在处理后重新组合。
    """
    def __init__(self, num_experts, gates):
        """Create a SparseDispatcher."""

        self._gates = gates  #表示每个样本对每个专家的权重
        self._num_experts = num_experts 

        # Jittor 中获取非零元素的索引 (N, 2) -> [batch_idx, expert_idx]
        nz = jt.nonzero(gates)
        expert_indices = nz[:, 1]
        sorted_indices = expert_indices.argsort()[1]

        # 记录排序后的 batch 索引和专家索引
        self._batch_index = nz[sorted_indices, 0].flatten()
        self._expert_index = nz[sorted_indices, 1].unsqueeze(1)

        # 计算每个专家分配到的样本数量 (用于后续 split)
        self._part_sizes = []
        for i in range(num_experts):
            self._part_sizes.append(int((nz[:, 1] == i).sum().item()))

        # 扩展 gates 以匹配 batch 索引，并提取对应的权重
        gates_exp = gates[self._batch_index]
        self._nonzero_gates = jt.gather(gates_exp, 1, self._expert_index)

    def dispatch(self, inp):
        """将输入 Tensor 分发给各个专家。"""

        # assigns samples to experts whose gate is nonzero

        # expand according to batch index so we can just split by _part_sizes
        inp_exp = inp[self._batch_index]
        return jt.split(inp_exp, self._part_sizes, dim=0)

    def combine(self, expert_out, multiply_by_gates=True):
        """
        将专家的输出按权重重新组合

        Sum together the expert output, weighted by the gates.
        The slice corresponding to a particular batch element `b` is computed
        as the sum over all experts `i` of the expert output, weighted by the
        corresponding gate values.  If `multiply_by_gates` is set to False, the
        gate values are ignored.
        """
        # apply exp to expert outputs, so we are not longer in log space
        stitched = jt.concat(expert_out, 0)

        if multiply_by_gates:
            stitched = stitched * self._nonzero_gates.unsqueeze(-1).unsqueeze(-1)

        # 创建全零容器，准备合并
        zeros = jt.zeros((self._gates.shape[0], expert_out[-1].shape[1], expert_out[-1].shape[2], expert_out[-1].shape[3]))
        
        # 使用 scatter(reduce='add') 替代 index_add
        # 需要扩展 batch index 以匹配 stitched 的形状
        idx = self._batch_index.reshape(-1, 1, 1, 1)
        idx = idx.expand_as(stitched)
        combined = zeros.scatter(0, idx, stitched, reduce='add')
        return combined
    
    def to_spatial(self, x, x_shape):
        """将频域特征转换回空间域。"""
        h, w = x_shape
        amp, phase = x.chunk(2, dim=1)
        real = amp * jt.cos(phase)
        imag = amp * jt.sin(phase)
        # Jittor 处理复数的方式：使用 jt.complex
        #x_complex = jt.complex(real, imag)
        # 逆傅里叶变换
        #x = jt.fft.ifft2(x_complex).real
        # Jittor 复数格式：最后一维 [real, imag]
        x_complex = jt.stack([real, imag], dim=-1)
        # 逆傅里叶变换 (ifft)
        x_ifft = nn._fft2(x_complex, True)
        x = x_ifft[..., 0]
        return x

    def expert_to_gates(self):
        """获取每个专家对应的门控值。"""
        # split nonzero gates for each expert
        return jt.split(self._nonzero_gates, self._part_sizes, dim=0)


##########################################################################
## Layer Norm 归一化层

def to_3d(x):
    # rearrange(x, 'b c h w -> b (h w) c')
    b, c, h, w = x.shape
    return x.permute(0, 2, 3, 1).reshape(b, h*w, c)

def to_4d(x,h,w):
    # rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)
    b, hw, c = x.shape
    return x.reshape(b, h, w, c).permute(0, 3, 1, 2)

class BiasFree_LayerNorm(nn.Module):
    """无偏置的层归一化。"""
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        
        # Jittor 中直接赋值 jt.Var 即可作为参数
        self.weight = jt.ones(normalized_shape)
        self.normalized_shape = normalized_shape

    def execute(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / jt.sqrt(sigma+1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    """带偏置的层归一化。"""
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        
        self.weight = jt.ones(normalized_shape)
        self.bias = jt.zeros(normalized_shape)
        self.normalized_shape = normalized_shape

    def execute(self, x):
        mu = x.mean(-1, keepdims=True)
        sigma = x.var(-1, keepdims=True)
        return (x - mu) / jt.sqrt(sigma+1e-5) * self.weight + self.bias

class LayerNorm(nn.Module):
    """通用的层归一化包装类，支持 4D 输入。"""
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        self.dim = dim
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def execute(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

class HighPassConv2d(nn.Module):
    """高通卷积：用于提取图像的高频细节（如边缘）。"""
    def __init__(self, c, freeze):
        super().__init__()
        
        self.conv = nn.Conv2d(
            in_channels=c, 
            out_channels=c, 
            kernel_size=3, 
            padding=1, 
            bias=False, 
            groups=c
        )
        
        # 使用 jt.array 定义初始卷积核
        kernel = jt.array([[[[-1, -1, -1],[-1, 8, -1],[-1, -1, -1]]]], dtype='float32')
        # 使用 assign 更新卷积权重
        self.conv.weight.assign(kernel.repeat(c, 1, 1, 1))
        
        if freeze:
            # 冻结权重梯度
            self.conv.weight.stop_grad()
        
    def execute(self, x):
        return self.conv(x)


##########################################################################
## Gated-Dconv Feed-Forward Network (GDFN) 门控深度卷积前馈网络 (GDFN)
'''
引入卷积：利用 3×3 卷积捕获图像的局部细节。
引入门控：利用双路乘法结构控制特征的传递，提升非线性表达能力。
'''

class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def execute(self, x):
        x = self.project_in(x)
        x1, x2 = jt.chunk(self.dwconv(x), 2, dim=1)
        x = nn.gelu(x1) * x2
        x = self.project_out(x)
        return x 
    
##########################################################################
## Multi-DConv Head Transposed Self-Attention 多头转置自注意力 (MDTA)
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        # Jittor 直接定义变量即可
        self.temperature = jt.ones((num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def execute(self, x):
        b,c,h,w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q,k,v = jt.chunk(qkv, 3, dim=1)   
        
        # q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        # k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        # v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        head = self.num_heads
        q = q.reshape(b, head, -1, h*w)
        k = k.reshape(b, head, -1, h*w)
        v = v.reshape(b, head, -1, h*w)

        q = jt.normalize(q, dim=-1)
        k = jt.normalize(k, dim=-1)

        # 使用 jt.matmul 进行矩阵乘法
        attn = (jt.matmul(q, k.transpose(-2, -1))) * self.temperature
        attn = nn.softmax(attn, dim=-1)

        out = jt.matmul(attn, v)
        
        # out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = out.reshape(b, -1, h, w)

        out = self.project_out(out)
        return out
    
class CrossAttention(nn.Module):
    """交叉注意力模块：用于融合两个不同来源的特征。"""
    def __init__(self, dim, num_heads, bias):
        super(CrossAttention, self).__init__()
        self.num_heads = num_heads
        self.temperature = jt.ones((num_heads, 1, 1))

        self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.q_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.kv = nn.Conv2d(dim, dim*2, kernel_size=1, bias=bias)
        self.kv_dwconv = nn.Conv2d(dim*2, dim*2, kernel_size=7, stride=1, padding=7//2, groups=dim*2, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        
    def execute(self, x, y):
        b,c,h,w = x.shape

        q = self.q_dwconv(self.q(x))
        kv = self.kv_dwconv(self.kv(y))
        k, v = jt.chunk(kv, 2, dim=1)
        
        # q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        # k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        # v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        head = self.num_heads
        q = q.reshape(b, head, -1, h*w)
        k = k.reshape(b, head, -1, h*w)
        v = v.reshape(b, head, -1, h*w)

        q = jt.normalize(q, dim=-1)
        k = jt.normalize(k, dim=-1)

        attn = (jt.matmul(q, k.transpose(-2, -1))) * self.temperature
        attn = nn.softmax(attn, dim=-1)

        out = jt.matmul(attn, v)
        
        # out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = out.reshape(b, -1, h, w)

        out = self.project_out(out)
        return out
    
##########################################################################
## Self-Attention in Fourier Domain  频域自注意力 (FFT Attention)
class FFTAttention(nn.Module):
    def __init__(self, dim: int, **kwargs):
        super(FFTAttention, self).__init__()

        self.patch_size = kwargs["patch_size"]
        
        self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        self.q_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)
        self.kv = nn.Conv2d(dim, dim*2, kernel_size=1, bias=False)
        self.kv_dwconv = nn.Conv2d(dim*2, dim*2, kernel_size=7, stride=1, padding=7//2, groups=dim*2)
        self.norm = LayerNorm(dim, "WithBias")
        self.proj_out = nn.Conv2d(dim, dim, kernel_size=1, padding=0)        
        
    def pad_and_rearrange(self, x):
        b, c, h, w = x.shape
        
        pad_h = (self.patch_size - (h % self.patch_size)) % self.patch_size
        pad_w = (self.patch_size - (w % self.patch_size)) % self.patch_size
        # Jittor 的 pad 使用列表 [left, right, top, bottom]
        x = nn.pad(x, [0, pad_w, 0, pad_h])
        # x = rearrange(x, 'b c (h p1) (w p2) -> b c h w p1 p2', p1=self.patch_size, p2=self.patch_size)
        H, W = x.shape[2], x.shape[3]
        p1 = self.patch_size
        p2 = self.patch_size
        h_ = H // p1
        w_ = W // p2
        x = x.reshape(b, c, h_, p1, w_, p2).permute(0, 1, 2, 4, 3, 5)
        return x
    
    def rearrange_to_original(self, x, x_shape):
        h, w = x_shape
        # x = rearrange(x, 'b c h w p1 p2 -> b c (h p1) (w p2)', p1=self.patch_size, p2=self.patch_size)
        b, c, h_, w_, p1, p2 = x.shape
        x = x.permute(0, 1, 2, 4, 3, 5).reshape(b, c, h_*p1, w_*p2)
        x = x[:, :, :h, :w]  # Slice out the original height and width
        return x

    def execute(self, x):
        b, c, h, w = x.shape

        q = self.q_dwconv(self.q(x))
        kv = self.kv_dwconv(self.kv(x))
        k, v = jt.chunk(kv, 2, dim=1)
            
        q = self.pad_and_rearrange(q)
        k = self.pad_and_rearrange(k)
            
        # Jittor 频域操作
        #q_fft = jt.fft.rfft2(q)
        #k_fft = jt.fft.rfft2(k)
        #out = q_fft * k_fft
        #out = jt.fft.irfft2(out)

        # Jittor 频域操作：构造复数并用 nn._fft2
        q_complex = jt.stack([q, jt.zeros_like(q)], dim=-1)
        k_complex = jt.stack([k, jt.zeros_like(k)], dim=-1)
        
        # Reshape to 4D for Jittor FFT
        B_dim, C_dim, H_blocks, W_blocks, P1, P2, _ = q_complex.shape
        q_complex_reshaped = q_complex.reshape(-1, P1, P2, 2)
        k_complex_reshaped = k_complex.reshape(-1, P1, P2, 2)
        
        q_fft = nn._fft2(q_complex_reshaped)
        k_fft = nn._fft2(k_complex_reshaped)
        
        # 复数乘法 (a+bi)(c+di) = (ac-bd) + (ad+bc)i
        q_real, q_imag = q_fft[..., 0], q_fft[..., 1]
        k_real, k_imag = k_fft[..., 0], k_fft[..., 1]
        out_real = q_real * k_real - q_imag * k_imag
        out_imag = q_real * k_imag + q_imag * k_real
        out_complex = jt.stack([out_real, out_imag], dim=-1)
        # 逆 FFT -> 取实部
        out = nn._fft2(out_complex, True)
        out = out[..., 0]
        
        # Reshape back to original 6D shape
        out = out.reshape(B_dim, C_dim, H_blocks, W_blocks, P1, P2)
        
        out = self.rearrange_to_original(out, (h, w))
        
        out = self.norm(out)
        out = out * v
        
        out = self.proj_out(out)
        return out

    
     
##########################################################################
## Adapter Block    专家适配器块 (ModExpert)
class ModExpert(nn.Module):
    def __init__(self, dim: int, rank: int, func: nn.Module, depth: int, patch_size: int, kernel_size:int):
        super(ModExpert, self).__init__()
        
        self.depth = depth
        self.proj = nn.ModuleList([
            nn.Conv2d(dim, rank, kernel_size=1, padding=0, bias=False),
            nn.Conv2d(dim, rank, kernel_size=1, padding=0, bias=False),
            nn.Conv2d(rank, dim, kernel_size=1, padding=0, bias=False)
        ])
        
        self.body = func(rank, kernel_size=kernel_size, patch_size=patch_size)
            
    def process(self, x, shared):
        shortcut = x
        x = self.proj[0](x)
        x = self.body(x) * nn.silu(self.proj[1](shared))
        x = self.proj[2](x)
        return x + shortcut

    def feat_extract(self, feats, shared):
        for _ in range(self.depth):
            feat = self.process(feats, shared)
        return feat
    
    def execute(self, x, shared):
        b, c, h, w = x.shape
        
        if b == 0:
            return x
        else:
            x = self.feat_extract(x, shared)
            return x
        



########################################################################### 
## Adapter Layer 适配器层 (Adapter Layer)
class AdapterLayer(nn.Module):
    def __init__(self, 
                 dim: int, rank: int, num_experts: int = 4, top_k: int=2, expert_layer: nn.Module=FFTAttention, stage_depth: int=1,
                 depth_type: str="lin", rank_type: str="constant", freq_dim: int=128, 
                 with_complexity: bool=False, complexity_scale: str="min"):
        super().__init__()            
        
        self.tau = 1
        self.loss = None
        self.top_k = top_k
        self.noise_eps = 1e-2
        self.num_experts = num_experts

        patch_sizes = [2**(i+2) for i in range(num_experts)]
        kernel_sizes = [3+(2*i) for i in range(num_experts)]
        
        if depth_type == "lin":
            depths = [stage_depth+i for i in range(num_experts)]
        elif depth_type == "double":
            depths = [stage_depth+(2*i) for i in range(num_experts)]
        elif depth_type == "exp":
            depths = [2**(i) for i in range(num_experts)]
        elif depth_type == "fact":
            depths = [math.factorial(i+1) for i in range(num_experts)]
        elif isinstance(depth_type, int):
            depths = [depth_type for _ in range(num_experts)]
        elif depth_type == "constant":
            depths = [stage_depth for i in range(num_experts)]
        else:
            raise(NotImplementedError)
        
        if rank_type == "constant":
            ranks = [rank for _ in range(num_experts)]
        elif rank_type == "lin":
            ranks = [rank+i for i in range(num_experts)]
        elif rank_type == "double":
            ranks = [rank+(2*i) for i in range(num_experts)]
        elif rank_type == "exp":
            ranks = [rank**(i+1) for i in range(num_experts)]
        elif rank_type == "fact":
            ranks = [math.factorial(rank+i) for i in range(num_experts)]
        elif rank_type == "spread":
            ranks = [dim//(2**i) for i in range(num_experts)][::-1]
        else:
            raise(NotImplementedError)
        
        self.experts = nn.ModuleList([
            MySequential(*[ModExpert(dim, rank=rank, func=expert_layer, depth=depth, patch_size=patch, kernel_size=kernel)])
            for idx, (depth, rank, patch, kernel) in enumerate(zip(depths, ranks, patch_sizes, kernel_sizes))
        ])
                
        self.proj_out = nn.Conv2d(dim, dim, kernel_size=1, padding=0, bias=False)
        # 使用 jt.array
        expert_complexity = jt.array([sum(p.numel() for p in expert.parameters()) for expert in self.experts])
        self.routing = RoutingFunction(
            dim, freq_dim, 
            num_experts=num_experts, k=top_k,
            complexity=expert_complexity, use_complexity_bias=with_complexity, complexity_scale=complexity_scale
        )
        
    def execute(self, x, freq_emb, shared):
        gates, top_k_indices, top_k_values, aux_loss = self.routing(x, freq_emb)
        self.loss = aux_loss
                
        # 路由分发
        if self.is_training():
            dispatcher = SparseDispatcher(self.num_experts, gates)
            expert_inputs = dispatcher.dispatch(x)
            expert_shared_intputs = dispatcher.dispatch(shared)
            expert_outputs = [self.experts[exp](expert_inputs[exp], expert_shared_intputs[exp]) for exp in range(len(self.experts))]
            out = dispatcher.combine(expert_outputs, multiply_by_gates=True)
        else:
            # 测试模式下的 Top-K 专家选择
            selected_experts = [self.experts[int(i.item())] for i in top_k_indices.squeeze(0)]
            expert_outputs = jt.stack([expert(x, shared) for expert in selected_experts], dim=1)
            gates = jt.gather(gates, 1, top_k_indices)  
            weighted_outputs = gates.unsqueeze(2).unsqueeze(3).unsqueeze(4) * expert_outputs 
            out = weighted_outputs.sum(1)
            
        out = self.proj_out(out)
        return out

##############################################################

class RoutingFunction(nn.Module):
    """路由函数：决定输入应该分配给哪些专家。"""
    def __init__(self, dim, freq_dim, num_experts, k, complexity, use_complexity_bias: bool = True, complexity_scale: str="max"):
        super(RoutingFunction, self).__init__()
        
        # 拆解 Sequential 以便手动处理维度变换
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.gate_linear = nn.Linear(dim, num_experts, bias=False)
        self.freq_gate = nn.Linear(freq_dim, num_experts, bias=False)
        
        if complexity_scale == "min":
            complexity = complexity / complexity.min()
        elif complexity_scale == "max":
            complexity = complexity / complexity.max()
        # Jittor 中直接赋值，使用 stop_grad 确保不参与梯度计算
        self.complexity = complexity.stop_grad()
        
        self.k = k
        self.tau = 1
        self.num_experts = num_experts
        self.noise_std = (1.0 / num_experts) * 1.0
        self.use_complexity_bias = use_complexity_bias

    def execute(self, x, freq_emb):
        b = x.shape[0]
        pooled = self.avg_pool(x).reshape(b, -1)
        logits = self.gate_linear(pooled) + self.freq_gate(freq_emb)
        if self.is_training():
            loss_imp = self.importance_loss(nn.softmax(logits, dim=-1))
        noise = jt.randn_like(logits) * self.noise_std
        noisy_logits = logits + noise

        gating_scores = nn.softmax(noisy_logits, dim=-1)
        top_k_values, top_k_indices = jt.topk(gating_scores, self.k, dim=-1)

        # Final auxiliary loss
        if self.is_training():
            loss_load = self.load_loss(logits, noisy_logits, self.noise_std)
            aux_loss = 0.5 * loss_imp + 0.5 * loss_load
        else:
            aux_loss = 0
        
        gates = jt.zeros_like(logits).scatter(1, top_k_indices, top_k_values)
        return gates, top_k_indices, top_k_values, aux_loss

    def importance_loss(self, gating_scores):
        importance = gating_scores.sum(0)
        importance = importance * (self.complexity * self.tau) if self.use_complexity_bias else importance
        imp_mean = importance.mean()
        imp_std = importance.std()
        loss_imp = (imp_std / (imp_mean + 1e-8)) ** 2
        return loss_imp

    def load_loss(self, logits, logits_noisy, noise_std):
        # Compute the noise threshold
        thresholds = jt.topk(logits_noisy, self.k, dim=-1)[1][:, -1]
        # Compute the load for each expert
        threshold_per_item = jt.sum(
            nn.one_hot(thresholds, self.num_experts) * logits_noisy,
            -1
        )
        
        # Calculate noise required to win
        noise_required_to_win = threshold_per_item.unsqueeze(-1) - logits
        noise_required_to_win /= noise_std
        
        # Probability of being above the threshold
        p = 1. - normal_cdf(noise_required_to_win)
        
        # Compute mean probability for each expert over examples
        p_mean = p.mean(0)
        
        # Compute p_mean's coefficient of variation squared
        p_mean_std = p_mean.std()
        p_mean_mean = p_mean.mean()
        loss_load = (p_mean_std / (p_mean_mean + 1e-8)) ** 2
        
        return loss_load




##########################################################################
## Encoder Block
class EncoderBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super().__init__()
        
        self.norms = nn.ModuleList([
          LayerNorm(dim, LayerNorm_type),
          LayerNorm(dim, LayerNorm_type)
        ])
        
        self.mixer = Attention(dim, num_heads, bias)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def execute(self, x):
        x = x + self.mixer(self.norms[0](x))
        x = x + self.ffn(self.norms[1](x))
        return x
        


##########################################################################
## Decoder Block
class DecoderBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type, expert_layer, complexity_scale=None,
                 rank=None, num_experts=None, top_k=None, depth_type=None, rank_type=None, stage_depth=None, freq_dim:int=128, with_complexity: bool=False):
        super().__init__()

        self.norms = nn.ModuleList([
          LayerNorm(dim, LayerNorm_type),
          LayerNorm(dim, LayerNorm_type),
        ])
        
        self.proj = nn.ModuleList([
            nn.Conv2d(dim, dim, kernel_size=1, padding=0),
            nn.Conv2d(dim, dim, kernel_size=1, padding=0)
        ])
        
        self.shared = Attention(dim, num_heads, bias)
        self.mixer = CrossAttention(dim, num_heads=num_heads, bias=bias)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)
        
        self.adapter = AdapterLayer(
            dim, rank, 
            top_k=top_k, num_experts=num_experts, expert_layer=expert_layer, freq_dim=freq_dim,
            depth_type=depth_type, rank_type=rank_type, stage_depth=stage_depth, 
            with_complexity=with_complexity, complexity_scale=complexity_scale
        )
        
    def execute(self, x, freq_emb=None):    
        shortcut = x
        x = self.norms[0](x)
        
        x_s = self.proj[0](x)
        x_a = self.proj[1](x)
        x_s = self.shared(x_s)
        x_a = self.adapter(x_a, freq_emb, x_s)
        x = self.mixer(x_a, x_s) + shortcut

        x = x + self.ffn(self.norms[1](x))
        return x, self.adapter.loss

    

######################################################################
## Encoder Residual Group
class EncoderResidualGroup(nn.Module):
    def __init__(self, 
                 dim: int, num_heads: List[int], num_blocks: int, ffn_expansion: int, LayerNorm_type: str, bias: bool):
        super().__init__()

        self.loss = None   
        self.num_blocks = num_blocks
        
        self.layers = nn.ModuleList([])
        for i in range(num_blocks):
            self.layers.append(
                EncoderBlock(dim, num_heads, ffn_expansion, bias, LayerNorm_type)
            )

    def execute(self, x):
        i = 0
        self.loss = 0
        while i < len(self.layers):
            x = self.layers[i](x)
            i += 1
        return x    
    
    
    
######################################################################
## Decoder Residual Group
class DecoderResidualGroup(nn.Module):
    def __init__(self, 
                 dim: int, num_heads: List[int], num_blocks: int, ffn_expansion: int, LayerNorm_type: str, bias: bool, complexity_scale=None,
                 rank=None, num_experts=None, expert_layer=None, top_k=None, depth_type=None, stage_depth=None, rank_type=None, freq_dim:int=128, with_complexity: bool=False):
        super().__init__()

        self.loss = None   
        self.num_blocks = num_blocks
        
        self.layers = nn.ModuleList([])
        for i in range(num_blocks):
            self.layers.append(
                DecoderBlock(
                    dim, num_heads, ffn_expansion, bias, LayerNorm_type, 
                    expert_layer=expert_layer, rank=rank, num_experts=num_experts, top_k=top_k, 
                    stage_depth=stage_depth, freq_dim=freq_dim, complexity_scale=complexity_scale,
                    depth_type=depth_type, rank_type=rank_type, with_complexity=with_complexity
                )
            )

    def execute(self, x, freq_emb=None):
        i = 0
        self.loss = 0
        while i < len(self.layers):
            x , loss = self.layers[i](x, freq_emb)
            self.loss += loss
            i += 1
        return x  
    
    
     
##########################################################################
## Overlapped image patch embedding with 3x3 Conv
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def execute(self, x):
        x = self.proj(x)
        return x

#new add
class PixelUnshuffle(nn.Module):
    def __init__(self, downscale_factor: int):
        super(PixelUnshuffle, self).__init__()
        self.r = downscale_factor

    def execute(self, x):
        b, c, h, w = x.shape
        r = self.r
        assert h % r == 0 and w % r == 0, "Height and Width must be divisible by downscale_factor"
        # return rearrange(x, 'b c (h r) (w r) -> b (c r r) h w', r=r)
        new_h, new_w = h // r, w // r
        x = x.reshape(b, c, new_h, r, new_w, r)
        x = x.permute(0, 1, 3, 5, 2, 4) # b c r r h w
        x = x.reshape(b, c*r*r, new_h, new_w)
        return x


##########################################################################
## Resizing modules 下采样
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat//2, kernel_size=3, stride=1, padding=1, bias=False),PixelUnshuffle(2))

    def execute(self, x):
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat*2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def execute(self, x):
        return self.body(x)
    
    
    
##########################################################################
## Frequency Embedding
class FrequencyEmbedding(nn.Module):
    """
    Embeds magnitude and phase features extracted from the bottleneck of the U-Net.
    """
    def __init__(self, dim):
        super(FrequencyEmbedding, self).__init__()
        self.high_conv = nn.Sequential(
            HighPassConv2d(dim, freeze=True),
            nn.GELU())
        
        self.mlp= nn.Sequential(
            nn.Linear(dim, 2*dim),
            nn.GELU(),
            nn.Linear(2*dim, dim)
            )

    def execute(self, x):
        x = self.high_conv(x)
        x = x.mean((-2, -1))
        x = self.mlp(x)
        return x
    
    
##########################################################################
##
class MoCEIR(nn.Module):
    def __init__(self,
                inp_channels=3, 
                out_channels=3, 
                dim = 32,
                levels: int = 4,
                heads = [1,1,1,1],
                num_blocks = [1,1,1,3],
                num_dec_blocks = [1, 1, 1],
                ffn_expansion_factor = 2,
                num_refinement_blocks = 1,
                LayerNorm_type = 'WithBias', ## Other option 'BiasFree'
                bias = False,
                rank=2,
                num_experts=4,
                depth_type="lin",
                stage_depth=[3,2,1],
                rank_type="constant",
                topk=1,
                expert_layer=FFTAttention,
                with_complexity=False,
                complexity_scale="max",
                ):
        super(MoCEIR, self).__init__()
        
        self.levels = levels
        self.num_blocks = num_blocks
        self.num_dec_blocks = num_dec_blocks
        self.num_refinement_blocks = num_refinement_blocks
        
        dims = [dim*2**i for i in range(levels)]
        ranks = [rank for i in range(levels-1)]

        # -- Patch Embedding
        self.patch_embed = OverlapPatchEmbed(in_c=inp_channels, embed_dim=dim, bias=False)
        self.freq_embed = FrequencyEmbedding(dims[-1])
                
        # -- Encoder --        
        self.enc = nn.ModuleList([])
        for i in range(levels-1):
            self.enc.append(nn.ModuleList([
                EncoderResidualGroup(
                    dim=dims[i], 
                    num_blocks=num_blocks[i], 
                    num_heads=heads[i],
                    ffn_expansion=ffn_expansion_factor, 
                    LayerNorm_type=LayerNorm_type, bias=True,),
                Downsample(dim*2**i)
                ])
            )
        
        # -- Latent --
        self.latent = EncoderResidualGroup(
            dim=dims[-1],
            num_blocks=num_blocks[-1], 
            num_heads=heads[-1], 
            ffn_expansion=ffn_expansion_factor,
            LayerNorm_type=LayerNorm_type, bias=True,)
                  
        # -- Decoder --
        dims = dims[::-1]
        ranks = ranks[::-1]
        heads = heads[::-1]
        num_dec_blocks = num_dec_blocks[::-1]
        
        self.dec = nn.ModuleList([])
        for i in range(levels-1):
            self.dec.append(nn.ModuleList([
                Upsample(dims[i]),
                nn.Conv2d(dims[i], dims[i+1], kernel_size=1, bias=bias),
                DecoderResidualGroup(
                    dim=dims[i+1],
                    num_blocks=num_dec_blocks[i], 
                    num_heads=heads[i+1],
                    ffn_expansion=ffn_expansion_factor, 
                    LayerNorm_type=LayerNorm_type, bias=bias, expert_layer=expert_layer, freq_dim=dims[0], with_complexity=with_complexity,
                    rank=ranks[i], num_experts=num_experts, stage_depth=stage_depth[i], depth_type=depth_type, rank_type=rank_type, top_k=topk, complexity_scale=complexity_scale),
                ])
            )

        # -- Refinement --
        heads = heads[::-1]
        self.refinement = EncoderResidualGroup(
            dim=dim,
            num_blocks=num_refinement_blocks, 
            num_heads=heads[0], 
            ffn_expansion=ffn_expansion_factor,
            LayerNorm_type=LayerNorm_type, bias=True,)
        
        self.output = nn.Conv2d(dim, out_channels, kernel_size=3, stride=1, padding=1, bias=bias)
        self.total_loss = None
    
    def execute(self, x, labels=None):
                
        feats = self.patch_embed(x)
        
        self.total_loss = 0
        enc_feats = []
        for i, (block, downsample) in enumerate(self.enc):
            feats = block(feats)
            enc_feats.append(feats)
            feats = downsample(feats)
        
        feats = self.latent(feats)
        freq_emb = self.freq_embed(feats)
                
        for i, (upsample, fusion, block) in enumerate(self.dec):
            feats = upsample(feats)
            feats = fusion(jt.concat([feats, enc_feats.pop()], dim=1))
            feats = block(feats, freq_emb)
            self.total_loss += block.loss

        feats = self.refinement(feats)
        x = self.output(feats) + x

        self.total_loss /= sum(self.num_dec_blocks)
        return x
    
                    
    
    
if __name__ == "__main__":
    # test
    model = MoCEIR(rank=2, num_blocks=[4,6,6,8], num_dec_blocks=[2,4,4], levels=4, dim=48, num_refinement_blocks=4, 
                   with_complexity=True, complexity_scale="max", stage_depth=[1,1,1], depth_type="constant", rank_type="spread", 
                   num_experts=4, topk=1, expert_layer=FFTAttention)

    x = jt.randn((1, 3, 224, 224))
    _ = model(x)
    print(model.total_loss)
    # Memory usage  
    # print('{:>16s} : {:<.3f} [M]'.format('Max Memery', torch.cuda.max_memory_allocated(torch.cuda.current_device())/1024**2))
    jt.display_memory_info()
  
    # FLOPS and PARAMS
    # flops = FlopCountAnalysis(model, (x))
    # print(flop_count_table(flops))