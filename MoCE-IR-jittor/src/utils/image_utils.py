"""
Created on 2020/9/8

@author: Boyun Li
"""
import os
import cv2
import numpy as np
import jittor as jt
import random
import jittor.nn as nn
from jittor import init
from PIL import Image
from .image_io import get_image_grid
import math

class EdgeComputation(nn.Module):
    def __init__(self, test=False):
        super(EdgeComputation, self).__init__()
        self.test = test
    def execute(self, x):
        if self.test:
            x_diffx = jt.abs(x[:, :, :, 1:] - x[:, :, :, :-1])
            x_diffy = jt.abs(x[:, :, 1:, :] - x[:, :, :-1, :])

            # y = torch.Tensor(x.size()).cuda()
            y = jt.zeros(x.shape)
            y[:, :, :, 1:] += x_diffx
            y[:, :, :, :-1] += x_diffx
            y[:, :, 1:, :] += x_diffy
            y[:, :, :-1, :] += x_diffy
            y = jt.sum(y, 1, keepdims=True) / 3
            y /= 4
            return y
        else:
            x_diffx = jt.abs(x[:, :, 1:] - x[:, :, :-1])
            x_diffy = jt.abs(x[:, 1:, :] - x[:, :-1, :])

            y = jt.zeros(x.shape)
            y[:, :, 1:] += x_diffx
            y[:, :, :-1] += x_diffx
            y[:, 1:, :] += x_diffy
            y[:, :-1, :] += x_diffy
            y = jt.sum(y, 0) / 3
            y /= 4
            return y.unsqueeze(0)


# randomly crop a patch from image
def crop_patch(im, pch_size):
    H = im.shape[0]
    W = im.shape[1]
    ind_H = random.randint(0, H - pch_size)
    ind_W = random.randint(0, W - pch_size)
    pch = im[ind_H:ind_H + pch_size, ind_W:ind_W + pch_size]
    return pch


# crop an image to the multiple of base
def crop_img(image, base=64):
    h = image.shape[0]
    w = image.shape[1]
    crop_h = h % base
    crop_w = w % base
    return image[crop_h // 2:h - crop_h + crop_h // 2, crop_w // 2:w - crop_w + crop_w // 2, :]


# image (H, W, C) -> patches (B, H, W, C)
def slice_image2patches(image, patch_size=64, overlap=0):
    assert image.shape[0] % patch_size == 0 and image.shape[1] % patch_size == 0
    H = image.shape[0]
    W = image.shape[1]
    patches = []
    image_padding = np.pad(image, ((overlap, overlap), (overlap, overlap), (0, 0)), mode='edge')
    for h in range(H // patch_size):
        for w in range(W // patch_size):
            idx_h = [h * patch_size, (h + 1) * patch_size + overlap]
            idx_w = [w * patch_size, (w + 1) * patch_size + overlap]
            patches.append(np.expand_dims(image_padding[idx_h[0]:idx_h[1], idx_w[0]:idx_w[1], :], axis=0))
    return np.concatenate(patches, axis=0)


# patches (B, H, W, C) -> image (H, W, C)
def splice_patches2image(patches, image_size, overlap=0):
    assert len(image_size) > 1
    assert patches.shape[-3] == patches.shape[-2]
    H = image_size[0]
    W = image_size[1]
    patch_size = patches.shape[-2] - overlap
    image = np.zeros(image_size)
    idx = 0
    for h in range(H // patch_size):
        for w in range(W // patch_size):
            image[h * patch_size:(h + 1) * patch_size, w * patch_size:(w + 1) * patch_size, :] = patches[idx,
                                                                                                 overlap:patch_size + overlap,
                                                                                                 overlap:patch_size + overlap,
                                                                                                 :]
            idx += 1
    return image


# def data_augmentation(image, mode):
#     if mode == 0:
#         # original
#         out = image.numpy()
#     elif mode == 1:
#         # flip up and down
#         out = np.flipud(image)
#     elif mode == 2:
#         # rotate counterwise 90 degree
#         out = np.rot90(image, axes=(1, 2))
#     elif mode == 3:
#         # rotate 90 degree and flip up and down
#         out = np.rot90(image, axes=(1, 2))
#         out = np.flipud(out)
#     elif mode == 4:
#         # rotate 180 degree
#         out = np.rot90(image, k=2, axes=(1, 2))
#     elif mode == 5:
#         # rotate 180 degree and flip
#         out = np.rot90(image, k=2, axes=(1, 2))
#         out = np.flipud(out)
#     elif mode == 6:
#         # rotate 270 degree
#         out = np.rot90(image, k=3, axes=(1, 2))
#     elif mode == 7:
#         # rotate 270 degree and flip
#         out = np.rot90(image, k=3, axes=(1, 2))
#         out = np.flipud(out)
#     else:
#         raise Exception('Invalid choice of image transformation')
#     return out

def data_augmentation(image, mode):
    if mode == 0:
        # original
        out = image.numpy()
    elif mode == 1:
        # flip up and down
        out = np.flipud(image)
    elif mode == 2:
        # rotate counterwise 90 degree
        out = np.rot90(image)
    elif mode == 3:
        # rotate 90 degree and flip up and down
        out = np.rot90(image)
        out = np.flipud(out)
    elif mode == 4:
        # rotate 180 degree
        out = np.rot90(image, k=2)
    elif mode == 5:
        # rotate 180 degree and flip
        out = np.rot90(image, k=2)
        out = np.flipud(out)
    elif mode == 6:
        # rotate 270 degree
        out = np.rot90(image, k=3)
    elif mode == 7:
        # rotate 270 degree and flip
        out = np.rot90(image, k=3)
        out = np.flipud(out)
    else:
        raise Exception('Invalid choice of image transformation')
    return out


# def random_augmentation(*args):
#     out = []
#     if random.randint(0, 1) == 1:
#         flag_aug = random.randint(1, 7)
#         for data in args:
#             out.append(data_augmentation(data, flag_aug).copy())
#     else:
#         for data in args:
#             out.append(data)
#     return out

def random_augmentation(*args):
    out = []
    flag_aug = random.randint(1, 7)
    for data in args:
        out.append(data_augmentation(data, flag_aug).copy())
    return out


def weights_init_normal_(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.uniform_(m.weight, 0.0, 0.02)
    elif classname.find('Linear') != -1:
        init.uniform_(m.weight, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        init.uniform_(m.weight, 1.0, 0.02)
        init.constant_(m.bias, 0.0)


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        # m.apply(weights_init_normal_)
        for submodule in m.modules():
            weights_init_normal_(submodule)
    elif classname.find('Linear') != -1:
        init.uniform_(m.weight, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        init.uniform_(m.weight, 1.0, 0.02)
        init.constant_(m.bias, 0.0)


def weights_init_xavier(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.xavier_gauss_(m.weight, gain=1)
    elif classname.find('Linear') != -1:
        init.xavier_gauss_(m.weight, gain=1)
    elif classname.find('BatchNorm2d') != -1:
        init.uniform_(m.weight, 1.0, 0.02)
        init.constant_(m.bias, 0.0)


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight, a=0, mode='fan_in')
    elif classname.find('BatchNorm2d') != -1:
        init.uniform_(m.weight, 1.0, 0.02)
        init.constant_(m.bias, 0.0)


def weights_init_orthogonal(m):
    classname = m.__class__.__name__
    print(classname)
    if classname.find('Conv') != -1:
        init.orthogonal_(m.weight, gain=1)
    elif classname.find('Linear') != -1:
        init.orthogonal_(m.weight, gain=1)
    elif classname.find('BatchNorm2d') != -1:
        init.uniform_(m.weight, 1.0, 0.02)
        init.constant_(m.bias, 0.0)


def init_weights(net, init_type='normal'):
    print('initialization method [%s]' % init_type)
    if init_type == 'normal':
        for m in net.modules():
            weights_init_normal(m)
    elif init_type == 'xavier':
        for m in net.modules():
            weights_init_xavier(m)
    elif init_type == 'kaiming':
        for m in net.modules():
            weights_init_kaiming(m)
    elif init_type == 'orthogonal':
        for m in net.modules():
            weights_init_orthogonal(m)
    else:
        raise NotImplementedError('initialization method [%s] is not implemented' % init_type)


def np_to_jt(img_np):
    """
    Converts image in numpy.array to jittor.Var.

    From C x W x H [0..1] to  C x W x H [0..1]

    :param img_np:
    :return:
    """
    return jt.array(img_np)[None, :]


def jt_to_np(img_var):
    """
    Converts an image in jittor.Var format to np.array.

    From 1 x C x W x H [0..1] to  C x W x H [0..1]
    :param img_var:
    :return:
    """
    return img_var.numpy()
    # return img_var.detach().cpu().numpy()[0]


def save_image(name, image_np, output_path="output/normal/"):
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    p = np_to_pil(image_np)
    p.save(output_path + "{}.png".format(name))


def np_to_pil(img_np):
    """
    Converts image in np.array format to PIL image.

    From C x W x H [0..1] to  W x H x C [0...255]
    :param img_np:
    :return:
    """
    ar = np.clip(img_np * 255, 0, 255).astype(np.uint8)

    if img_np.shape[0] == 1:
        ar = ar[0]
    else:
        assert img_np.shape[0] == 3, img_np.shape
        ar = ar.transpose(1, 2, 0)

    return Image.fromarray(ar)


def tensor2img(tensor, rgb2bgr=True, out_type=np.uint8, min_max=(0, 1)):
    """Convert jittor Vars into image numpy arrays.

    After clamping to [min, max], values will be normalized to [0, 1].

    Args:
        tensor (Var or list[Var]): Accept shapes:
            1) 4D mini-batch Var of shape (B x 3/1 x H x W);
            2) 3D Var of shape (3/1 x H x W);
            3) 2D Var of shape (H x W).
            Var channel should be in RGB order.
        rgb2bgr (bool): Whether to change rgb to bgr.
        out_type (numpy type): output types. If ``np.uint8``, transform outputs
            to uint8 type with range [0, 255]; otherwise, float type with
            range [0, 1]. Default: ``np.uint8``.
        min_max (tuple[int]): min and max values for clamp.

    Returns:
        (Var or list): 3D ndarray of shape (H x W x C) OR 2D ndarray of
        shape (H x W). The channel order is BGR.
    """
    if not (isinstance(tensor, jt.Var) or (isinstance(tensor, list) and all(isinstance(t, jt.Var) for t in tensor))):
        raise TypeError(f'tensor or list of tensors expected, got {type(tensor)}')

    if isinstance(tensor, jt.Var):
        tensor = [tensor]
    result = []
    for _tensor in tensor:
        _tensor = _tensor.squeeze(0).float().clamp(min_max[0], min_max[1])
        _tensor = (_tensor - min_max[0]) / (min_max[1] - min_max[0])

        n_dim = _tensor.ndim
        if n_dim == 4:
            # Convert to numpy (B, C, H, W)
            _numpy = _tensor.numpy()
            # Convert to list of (C, H, W)
            img_list = [_numpy[i] for i in range(_numpy.shape[0])]
            # Make grid: returns (C, GridH, GridW)
            img_np = get_image_grid(img_list, nrow=int(math.sqrt(_tensor.shape[0])))
            img_np = img_np.transpose(1, 2, 0)
            if rgb2bgr:
                img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        elif n_dim == 3:
            img_np = _tensor.numpy()
            img_np = img_np.transpose(1, 2, 0)
            if img_np.shape[2] == 1:  # gray image
                img_np = np.squeeze(img_np, axis=2)
            else:
                if rgb2bgr:
                    img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        elif n_dim == 2:
            img_np = _tensor.numpy()
        else:
            raise TypeError(f'Only support 4D, 3D or 2D tensor. But received with dimension: {n_dim}')
        if out_type == np.uint8:
            # Unlike MATLAB, numpy.unit8() WILL NOT round by default.
            img_np = (img_np * 255.0).round()
        img_np = img_np.astype(out_type)
        result.append(img_np)
    if len(result) == 1:
        result = result[0]
    return result