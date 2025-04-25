# 1. process train data, then test data.
import os

from PIL import Image
import torchvision.transforms.functional as transform_F
import torch
import torch.nn.functional as F
import math
import cv2
import numpy as np

def generate_ams(imgs4fft, save_path):
    ams_save_path = save_path + '.png'
    if os.path.isfile(ams_save_path):
        print('ams_save_path is already exists, pass')
        return
    imgs4fft = imgs4fft[None,:,:,:]
    for j in range(imgs4fft.shape[0]):
        img_h, img_w = imgs4fft[j].shape[1], imgs4fft[j].shape[2]
        # size of the patch which used to compute the amplitude spectrum
        num_grid_h = 128
        ratio = img_w / img_h
        num_grid_w = int(num_grid_h * ratio)
        # number pixels per patch
        pix_per_grid_h, pix_per_grid_w = math.ceil(img_h / num_grid_h), math.ceil(img_w / num_grid_w)
        target_h, target_w = (pix_per_grid_h * num_grid_h), (pix_per_grid_w * num_grid_w)
        interpolated_samples = F.interpolate(imgs4fft, size=(target_h, target_w), mode='bilinear',)
        tmp_am = []
        for k in range(num_grid_h):
            for p in range(num_grid_w):
                normed_img = interpolated_samples[j]
                patch = normed_img[:, k*pix_per_grid_h:(k+1)*pix_per_grid_h, p*pix_per_grid_w:(p+1)*pix_per_grid_w]
                spec_patch = torch.fft.rfft2(patch, norm='ortho')
                am_patch = torch.abs(spec_patch)
                tmp_am.append(am_patch.sum().unsqueeze(dim=0))
        am_map = torch.cat(tmp_am, dim=0).reshape(num_grid_h, num_grid_w).unsqueeze(dim=0).unsqueeze(dim=0)  # 1，1，16，32
        interpolated_am_map = F.interpolate(am_map, size=(img_h, img_w), mode='bilinear',)
        np_am_map = interpolated_am_map.squeeze(0).numpy().transpose(1, 2, 0)
        image_cv_trans = cv2.cvtColor((np_am_map / np_am_map.max() * 255).astype(np.uint8), cv2.COLOR_BGR2RGB)
        resized_image_trans = cv2.resize(image_cv_trans, (img_w, img_h))
        cv2.imwrite('{}'.format(ams_save_path), resized_image_trans)
        print("saved to " + ams_save_path)

def main():
    root = '/media/pj/UPC1/TIP-datasets-for-github/samples/'
    image_path = '/media/pj/UPC1/TIP-datasets-for-github/samples/images/'
    filenames = os.listdir(image_path)

    for i, img_name in enumerate(filenames):
        image = Image.open(os.path.join(image_path, img_name)).convert('RGB')
        image_tensor = transform_F.to_tensor(image)
        image_tensor = transform_F.normalize(image_tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        am_dir = root + 'ams_img'
        if os.path.isdir(am_dir):
            pass
        else:
            os.makedirs(am_dir)
        current_img_name_no_postfix = img_name.split('.png')[0]
        current_am_path = am_dir + '/' + current_img_name_no_postfix
        generate_ams(image_tensor, current_am_path)
        print(f'total images : {len(filenames)}, current is {i+1}')

if __name__ == '__main__':
    main()
