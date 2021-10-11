import PIL
import torch
from torchvision.transforms.functional import adjust_brightness as torch_adjust_brightness, pil_to_tensor
from torchvision.transforms.functional import adjust_contrast as torch_adjust_contrast
from torchvision.transforms.functional import adjust_gamma as torch_adjust_gamma
from torchvision.transforms.functional import rotate as torch_rotate
from torchvision.transforms.functional import crop as torch_crop
from torchvision.transforms.functional import affine as torch_affine
from torchvision.transforms.functional import hflip as torch_hflip, vflip as torch_vlip
from torchvision.transforms.functional import resize as torch_resize
from torchvision.transforms.functional import normalize as torch_norm
from torchvision.transforms.functional import gaussian_blur as torch_gaussian_blur
from torchvision.transforms.functional import F_t
from torchvision.transforms import RandomAffine, InterpolationMode
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

import random 

from PIL import ImageFilter, Image

from timeit import default_timer as timer

def adjust_brightness_tensor(image, min_brightness_ratio = 0.5, max_brightness_ratio = 1.5, variance = 0.5):
    new_image = []
    for im in image:
        if np.random.uniform(0,1) < 0.5:
            brightness_scale = np.random.normal(1, variance)
            brightness_scale = np.clip(brightness_scale, min_brightness_ratio, max_brightness_ratio)  
            im = torch_adjust_brightness(im, brightness_scale)
        new_image.append(im)
    return new_image

def adjust_brightness_PIL(image, min_brightness_ratio = 0.5, max_brightness_ratio = 1.5, variance = 0.5):
    brightness_scale = np.random.normal(1, variance)
    brightness_scale = np.clip(brightness_scale, min_brightness_ratio, max_brightness_ratio)  
    image = torch_adjust_brightness(image, brightness_scale)
    return image

def adjust_contrast_tensor(image, min_contrast_ratio = 0.5, max_contrast_ratio = 1.5, variance = 0.5):
    for im in image:
        contrast_scale = np.random.normal(1, variance)
        contrast_scale = np.clip(contrast_scale, min_contrast_ratio, max_contrast_ratio)
        image = torch.cat((image, torch.zeros_like(image[0,:,:].unsqueeze(0))), dim=0)
        image = torch_adjust_contrast(image, contrast_scale)
        return image[:2, :, :]
        
def adjust_contrast_PIL(image, min_contrast_ratio = 0.5, max_contrast_ratio = 1.5, variance = 0.5):
    contrast_scale = np.random.normal(1, variance)
    contrast_scale = np.clip(contrast_scale, min_contrast_ratio, max_contrast_ratio)
    image = torch_adjust_contrast(image, contrast_scale)
    return image

def adjust_gamma_tensor(image, min_gamma = 0.5, max_gamma = 2, variance = 0.5):
    new_image = []
    for im in image:
        if np.random.uniform(0,1) < 0.5:
            normal_sample = np.random.normal(0, variance)
            if normal_sample > 0:
                gamma = 1 + normal_sample
            else:
                gamma = 1.0 / (1 + np.abs(normal_sample))
            gamma = np.clip(gamma, min_gamma, max_gamma)
            im = torch_adjust_gamma(im, gamma)
        new_image.append(im)
    return new_image 

def adjust_gamma_PIL(image, min_gamma = 0.5, max_gamma = 2, variance = 0.5):
    normal_sample = np.random.normal(0, variance)
    if normal_sample > 0:
        gamma = 1 + normal_sample
    else:
        gamma = 1.0 / (1 + np.abs(normal_sample))
    gamma = np.clip(gamma, min_gamma, max_gamma)
    return torch_adjust_gamma(image, gamma)

def rotate_90(image, mask):
    rotate_angle = np.random.randint(1, 4) * 90
    image = torch_rotate(image, rotate_angle)
    mask = torch_rotate(mask, rotate_angle)
    return image, mask

def hflip(image, mask):
    if isinstance(image, torch.Tensor):
        return image.flip(-1), mask.flip(-1)
    else:
        return torch_hflip(image), torch_hflip(mask)

def vflip(image, mask):
    image = torch_vflip(image)
    mask = torch_vflip(mask)
    return image, mask

def resize(image, mask, image_dim, int_im=InterpolationMode.BILINEAR, int_mask=InterpolationMode.BILINEAR):
    image = torch_resize(image, image_dim, interpolation=int_im)
    mask = torch_resize(mask, image_dim, interpolation=int_mask)
    return image, mask

def low_res(image):
    new_image = []
    for im in image:
        image_dim = [im.shape[-2], im.shape[-1]]
        res_factor = np.random.uniform(1,2)
        new_res = int(image_dim[0] / res_factor)
        im = torch_resize(im, [new_res, new_res], InterpolationMode.NEAREST)             
        im = torch_resize(im, image_dim, InterpolationMode.BICUBIC)  
        new_image.append(im)
    return new_image

def resize_double(image, image2, mask, image_dim):
    image = torch_resize(image, image_dim)
    image2 = torch_resize(image2, image_dim)
    mask = torch_resize(mask, image_dim)
    return image, image2, mask

def random_square_crop_by_scale(image, mask, scale = 0.9):
    if isinstance(image, torch.Tensor):
        old_size = image.shape[-1]
    else:
        old_size = image.size[-1]
    new_size = int(old_size * scale)
    start_x = np.random.randint(0, old_size - new_size)
    start_y = np.random.randint(0, old_size - new_size)
    image_cropped = image[:, start_x: start_x + new_size, start_y: start_y + new_size]
    mask_cropped = mask[:, start_x: start_x + new_size, start_y: start_y + new_size]
    return image_cropped, mask_cropped

def gaussian_blur(image, radius = 5):
    new_image=[]
    for im in image:
        if np.random.uniform(0,1) < 0.5:
            sigma = np.random.uniform(0.5, 1.5)
            im = torch_gaussian_blur(im, radius, sigma=sigma)
        new_image.append(im)
    return new_image

def shear(image, mask, shear_degrees):
    affine = RandomAffine(degrees = 0,shear = shear_degrees)
    seed = int(timer())
    
    random.seed(seed)
    image = affine(image)
    random.seed(seed)
    mask = affine(mask)
    
    return image, mask

def shift(image, mask, max_shift_h = 20, max_shift_v = 20):
    shift1 = np.random.randint(-max_shift_h, max_shift_h)
    shift2 = np.random.randint(-max_shift_v, max_shift_v)
    
    a = 1
    b = 0
    c = shift1
    d = 0
    e = 1
    f = shift2
    
    image = torch_affine(image, 0.0, [c, f], 1.0, 0.0)
    mask = torch_affine(mask, 0.0, [c, f], 1.0, 0.0)
    
    return image, mask

def rotate(image, mask, max_angle):
    angle = np.random.uniform(-max_angle, max_angle)
    return torch_rotate(image, angle), torch_rotate(mask, angle)

def scale(image, mask, variance = 0.1, min_scale = 0.7, max_scale = 1.3):
    if isinstance(image, torch.Tensor):
        old_size = image.shape[-1]
    else:
        old_size = image.size[-1]
    
    new_size = int(np.random.normal(old_size, old_size * 0.1))
    new_size = np.clip(new_size, old_size * min_scale, old_size * max_scale)
    #print new_size
    
    if new_size < old_size:
        left = int(np.random.uniform(0, old_size-new_size) )
        upper = int(np.random.uniform(0, old_size-new_size))
    else:
        left, upper = 0,0
    # low = int(upper + new_size)
    # right = int(left + new_size)

    new_size = int(new_size)
    
    return torch_crop(image, upper, left, new_size, new_size), torch_crop(mask, upper, left, new_size, new_size)
    # return image.crop((left, upper, right, low)), mask.crop((left, upper, right, low))

def normalize(image):
    return torch_norm(image, mean=[0.31, 0.20], std=[0.24, 0.28])

def channel_dropout(image):
    R, G, B = 1, 1, 0
    channel = np.random.randint(2)
    if channel == 0:
        R = 0
    else:
        G = 0 

    matrix = ( R, 0, 0, 0,
                0, G, 0, 0,
                0, 0, B, 0)

    return image.convert("RGB", matrix) 
