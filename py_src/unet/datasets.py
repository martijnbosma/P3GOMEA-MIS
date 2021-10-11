import torch
import torchvision
from PIL import Image
import os
import numpy as np
from unet.augmentations import *
import torch.utils.data as data
class DoubleImageDataset(data.Dataset):
    
    def __init__(self, dir_images_list, dir_masks_list, patients_list, batch_size=16, augment=False, image_dim=(256, 256), num_classes=2):
        
        self.all_patients_masks = []
        self.all_patients_images = []
        self.all_patients_images_filenames = []
        self.all_patients_masks_filenames = []
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.augment = augment
        self.image_dim = image_dim
        self.to_tensor = torchvision.transforms.ToTensor()
        self.to_image = torchvision.transforms.ToPILImage()

        for dir_image, dir_mask, patients in zip(dir_images_list, dir_masks_list, patients_list):
            images = os.listdir(dir_image)
            masks = os.listdir(dir_mask)

            for image_filename in images:
                
                if os.path.exists(os.path.join(dir_image, image_filename)):
                    patient = int(image_filename.split('_')[0].replace('Case',''))
                    if patient in patients:
                        self.all_patients_images_filenames.append(os.path.join(dir_image, image_filename))
                        self.all_patients_masks_filenames.append(os.path.join(dir_mask, image_filename))

        np.random.seed(42)        
        ind = np.random.permutation(len(self.all_patients_images_filenames))
        self.all_patients_images_filenames = np.array(self.all_patients_images_filenames)[ind]
        self.all_patients_masks_filenames = np.array(self.all_patients_masks_filenames)[ind]
        
        for image_filename, mask_filename in zip(self.all_patients_images_filenames, self.all_patients_masks_filenames):
            image = Image.open(image_filename)
            mask = Image.open(mask_filename)

            image, mask = resize(image, mask, self.image_dim,
                        int_im=InterpolationMode.BICUBIC, int_mask=InterpolationMode.BICUBIC)

            image_tensor, mask_tensor = self.to_tensor(image), self.to_tensor(mask)

            image.close()
            mask.close()

            if image_tensor.shape[0] > 1: 
                    image_tensor = image_tensor[:2,:,:]

            if mask_tensor.shape[0] > 1:
                mask_tensor = torch.argmax(mask_tensor, dim=0).unsqueeze(0).float()
            else:
                mask_tensor = (mask_tensor >= 0.5).float()

            self.all_patients_images.append(image_tensor)            
            self.all_patients_masks.append(mask_tensor)

        self.all_patients_images = tuple(self.all_patients_images)
        self.all_patients_masks = tuple(self.all_patients_masks)

    def __len__(self):
        return len(self.all_patients_images)

    def __getitem__(self, index):
        image_filename, mask_filename = self.all_patients_images_filenames[
            index], self.all_patients_masks_filenames[index]
        image, mask = self.all_patients_images[index].detach().clone(), self.all_patients_masks[index].detach().clone()

        if self.augment:
            
            if np.random.uniform(0,1) < 0.3:
                image, mask = scale(image, mask, variance = 0.2, min_scale = 0.7, max_scale = 1.3)
            
            if np.random.uniform(0,1) < 0.3:            
                image, mask = shift(image, mask, max_shift_h = 40, max_shift_v = 40)
            
            if np.random.uniform(0,1) < 0.3:            
                image, mask = rotate(image, mask, 10)     
            
            # if np.random.uniform(0,1) < 0.4:            
            #     image, mask = hflip(image, mask)
            
            #------- per channel operations --------

            if image.shape[0] > 1:
                image = [image[0].unsqueeze(0), image[1].unsqueeze(0)]
            else:
                image = [image]

            if np.random.uniform(0,1) < 0.3:                   
                image = adjust_brightness_tensor(image, 0.5, 1.5, variance = 0.5)

            # if np.random.uniform(0,1) < 0.3:            
            #     image = adjust_gamma_tensor(image, min_gamma=0.7, max_gamma=1.5, variance=0.5)

            # if np.random.uniform(0,1) < 0.3:                   
            #     image = adjust_contrast_tensor(image, 0.8, 1.25, variance = 0.25)
            
            # if np.random.uniform(0,1) < 0.2:                   
            #     image = gaussian_blur(image)            
            
            # if np.random.uniform(0,1) < 0.15:
            #     image = low_res(image)

            if len(image) > 1:
                image = torch.cat((image[0], image[1]), dim=0)
            else:
                image = image[0]

        if image.shape[-1] != self.image_dim[-1] or image.shape[-2] != self.image_dim[-2] \
                or mask.shape[-1] != self.image_dim[-1] or mask.shape[-2] != self.image_dim[-2]:
            image, mask = resize(image, mask, self.image_dim,
                            int_im=InterpolationMode.BILINEAR, int_mask=InterpolationMode.NEAREST)

        return image.detach(), mask.detach(), image_filename.split('/')[-1]

class DoubleImageDatasetPIL(data.Dataset):
    
    def __init__(self, dir_images_list, dir_masks_list, patients_list, batch_size=16, augment=False, image_dim=(256, 256), num_classes=2):
        
        self.all_patients_masks = []
        self.all_patients_images = []
        self.all_patients_images_filenames = []
        self.all_patients_masks_filenames = []
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.augment = augment
        self.image_dim = image_dim
        self.to_tensor = torchvision.transforms.ToTensor()
        self.to_image = torchvision.transforms.ToPILImage()

        for dir_image, dir_mask, patients in zip(dir_images_list, dir_masks_list, patients_list):
            images = os.listdir(dir_image)
            masks = os.listdir(dir_mask)

            for image_filename in images:
                
                if os.path.exists(os.path.join(dir_image, image_filename)):
                    patient = int(image_filename.split('_')[0].replace('Case',''))
                    if patient in patients:
                        self.all_patients_images_filenames.append(os.path.join(dir_image, image_filename))
                        self.all_patients_masks_filenames.append(os.path.join(dir_mask, image_filename))

        np.random.seed(42)        
        ind = np.random.permutation(len(self.all_patients_images_filenames))
        self.all_patients_images_filenames = np.array(self.all_patients_images_filenames)[ind]
        self.all_patients_masks_filenames = np.array(self.all_patients_masks_filenames)[ind]
        
        for image_filename, mask_filename in zip(self.all_patients_images_filenames, self.all_patients_masks_filenames):
            image = Image.open(image_filename)
            mask = Image.open(mask_filename)

            image, mask = resize(image, mask, self.image_dim,
                        int_im=InterpolationMode.BICUBIC, int_mask=InterpolationMode.NEAREST)

            self.all_patients_images.append(image.copy())            
            self.all_patients_masks.append(mask.copy())

            image.close()
            mask.close()

    def __len__(self):
        return len(self.all_patients_images)

    def __getitem__(self, index):
        image_filename, mask_filename = self.all_patients_images_filenames[
            index], self.all_patients_masks_filenames[index]
        image, mask = self.all_patients_images[index], self.all_patients_masks[index]

        if self.augment:
            
            if np.random.uniform(0,1) < 0.3:
                image, mask = scale(image, mask, variance = 0.2, min_scale = 0.8, max_scale = 1.3)
            
            if np.random.uniform(0,1) < 0.3:            
                image, mask = shift(image, mask, max_shift_h = 5, max_shift_v = 5)
            
            if np.random.uniform(0,1) < 0.3:            
                image, mask = rotate(image, mask, 5)      
            
            if np.random.uniform(0,1) < 0.3:                   
                image = adjust_brightness(image, 0.7, 1.4, variance = 0.25)
            
            if np.random.uniform(0,1) < 0.45:            
                image, mask = hflip(image, mask)   

            if np.random.uniform(0,1) < 0.2:                   
                image = gaussian_blur(image)
            
            # if np.random.uniform(0,1) < 0.2:                   
            #     image = adjust_brightness(image, 0.7, 1.4, variance=0.25)

            # if np.random.uniform(0,1) < 0.1:                   
            #     image = mask_transform(image, 0.7, 1.5, variance=0.3)
            
            # if np.random.uniform(0,1) < 0.3:            
            #     image = adjust_gamma(image, min_gamma=0.7, max_gamma=1.4, variance=0.25)

            # if image.mode == 'RGB':
            #     if np.random.uniform(0,1) < 0.2:            
            #         image = adjust_contrast(image, 0.7, 1.4, variance = 0.25)
            
            #     if np.random.uniform(0,1) < 0.2:                   
            #         image = channel_dropout(image)

        image, mask = resize(image, mask, self.image_dim,
                        int_im=InterpolationMode.BICUBIC, int_mask=InterpolationMode.NEAREST)

        image, mask = self.to_tensor(image), self.to_tensor(mask)

        if image.shape[0] > 1: 
            image = image[:2,:,:]
            # if np.random.uniform(0,1) < 0.3:                   
            #     image = normalize(image) 
      
        if mask.shape[0] > 1:
            mask = torch.argmax(mask, dim=0).unsqueeze(0).float()

        return image, mask, image_filename.split('/')[-1]


class ImageDataset(data.Dataset):
    
    def __init__(self, dir_images_list, dir_masks_list, patients_list, batch_size=16, augment=False, image_dim=(256, 256)):
        
        self.all_patients_masks = []
        self.all_patients_images = []
        self.all_patients_images_filenames = []
        self.all_patients_masks_filenames = []
        self.batch_size = batch_size

        for dir_image, dir_mask, patients in zip(dir_images_list, dir_masks_list, patients_list):
            images = os.listdir(dir_image)
            masks = os.listdir(dir_mask)

            for mask_filename in masks:
                
                if os.path.exists(os.path.join(dir_image, mask_filename)):
                    patient = int(mask_filename.split('_')[0].replace('Case',''))
                    if patient in patients:
                        self.all_patients_images_filenames.append(os.path.join(dir_image, mask_filename))
                        self.all_patients_masks_filenames.append(os.path.join(dir_mask, mask_filename))
                    
        self.augment = augment
        self.image_dim = image_dim

        np.random.seed(42)        
        ind = np.random.permutation(len(self.all_patients_images_filenames))
        self.all_patients_images_filenames = np.array(self.all_patients_images_filenames)[ind]
        self.all_patients_masks_filenames = np.array(self.all_patients_masks_filenames)[ind]
        
        for image_filename, mask_filename in zip(self.all_patients_images_filenames, self.all_patients_masks_filenames):
            image = Image.open(image_filename)
            mask = Image.open(mask_filename)

            image, mask = resize(image, mask, self.image_dim)

            self.all_patients_images.append(image.copy())            
            self.all_patients_masks.append(mask.copy())
            
            image.close()
            mask.close()

        self.to_tensor = torchvision.transforms.ToTensor()
        self.to_image = torchvision.transforms.ToPILImage()

    def __len__(self):
        return len(self.all_patients_images)

    def __getitem__(self, index):
        image_filename, mask_filename = self.all_patients_images_filenames[
            index], self.all_patients_masks_filenames[index]
        image, mask = self.all_patients_images[index], self.all_patients_masks[index]
        
        if self.augment:
            
            if np.random.uniform(0,1) < 0.3:
                image, mask = scale(image, mask, variance = 0.2, min_scale = 0.7, max_scale = 1.3)
            
            if np.random.uniform(0,1) < 0.3:            
                image, mask = shift(image, mask, max_shift_h = 40, max_shift_v = 40)
            
            if np.random.uniform(0,1) < 0.3:            
                image, mask = rotate(image, mask, 10)            
            
            if np.random.uniform(0,1) < 0.3:                   
                image = adjust_brightness_PIL(image, 0.5, 1.5, variance = 0.5)
            
            if np.random.uniform(0,1) < 0.3:            
                image = adjust_contrast_PIL(image, 0.5, 1.5, variance = 0.5)            

        if image.size[0] != self.image_dim[0] or image.size[1] != self.image_dim[1]:
            image, mask = resize(image, mask, self.image_dim)
        
        image, mask = self.to_tensor(image), self.to_tensor(mask)
        
        mask = (mask >= 0.5).float()
        
        return image, mask, image_filename.split('/')[-1]