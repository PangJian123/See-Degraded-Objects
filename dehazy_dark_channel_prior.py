import os
import os.path as osp

import cv2
from PIL import Image
from torchvision.transforms import functional as F
from network_files.det_utils import DarkChannelPrior
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import transforms
dark_channel_piror = DarkChannelPrior(kernel_size=15, top_candidates_ratio=0.0001,
                                          omega=0.95, radius=40, eps=1e-3, open_threshold=True, depth_est=True)


image_path = '/media/pj/UPC/samples/images/'
dehazy_transmission_path = '/media/pj/UPC/samples/transmission/'
# dehazy_image_path = '/media/pj/UPC/samples/dehazy_images/'
# dehazy_depth_path = '/media/pj/UPC/samples/depth/'

files = os.listdir(image_path)
for i, file_name in enumerate(files):
    img = cv2.imread(image_path + file_name)
    img_height = img.shape[0]
    img_width = img.shape[1]
    # # To prevent out of memory
    # if img_height>=1024 or img_width>=2048:
    #     print(f"{file_name} is too big with size {img.shape}")
    #     img = cv2.imread(image_path + file_name)
    #     img = cv2.resize(img, (int(img_width/2), int(img_height/2)))
    #     print(f"resize to {img.shape}")

    img = np.array(img)
    img_data = np.asarray(img, dtype=np.float64)/255
    img_tensor = torch.from_numpy(img_data)
    img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)
    dehaze_images_tensor, dc, airlight, raw_t, refined_transmission, depth = dark_channel_piror(img_tensor)
    dehazy_img_np = dehaze_images_tensor.squeeze(0).numpy().transpose(1, 2, 0)
    refined_transmission_img_np = refined_transmission.squeeze(0).numpy().transpose(1, 2, 0)
    depth_img_np = depth.squeeze(0).numpy().transpose(1, 2, 0)

    # # # -----------------------save dehazy image
    # if dehazy_img_np.shape[0] != img_height or dehazy_img_np.shape[1] != img_width:
    #     image_cv = cv2.cvtColor((dehazy_img_np * 255).astype(np.uint8), cv2.COLOR_BGR2RGB)
    #     resized_image = cv2.resize(image_cv, (img_width, img_height))
    #     BGR2RGB_image_cv = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
    #     cv2.imwrite('{}{}'.format(dehazy_image_path, file_name), BGR2RGB_image_cv)
    #     print(f"size of generated image is not equal to the orinal size, resized image, generate {i} images,total {len(files)} images")
    # else:
    #     cv2.imwrite('{}{}'.format(dehazy_image_path, file_name), dehazy_img_np * 255)
    #     print(f"generate {i} images,total {len(files)} images")

    # -----------------------save transmission map
    if refined_transmission_img_np.shape[0] != img_height or refined_transmission_img_np.shape[1] != img_width:
        image_cv_trans = cv2.cvtColor((refined_transmission_img_np * 255).astype(np.uint8), cv2.COLOR_BGR2RGB)
        resized_image_trans = cv2.resize(image_cv_trans, (img_width, img_height))
        cv2.imwrite('{}{}'.format(dehazy_transmission_path, file_name), resized_image_trans)
    else:
        cv2.imwrite('{}{}'.format(dehazy_transmission_path, file_name), refined_transmission_img_np * 255)
        print(f"generate {i} transmissions,total {len(files)} transmissions")

    # # -----------------------save depth
    # if depth_img_np.shape[0] != img_height or depth_img_np.shape[1] != img_width:
    #     image_cv_depth = cv2.cvtColor((depth_img_np * 255 / depth_img_np.max()).astype(np.uint8), cv2.COLOR_BGR2RGB)
    #     resized_image_trans = cv2.resize(image_cv_depth, (img_width, img_height))
    #     cv2.imwrite('{}{}'.format(dehazy_depth_path, file_name), resized_image_trans)
    # else:
    #     cv2.imwrite('{}{}'.format(dehazy_depth_path, file_name), depth_img_np * 255 / depth_img_np.max())
    #     print(f"generate {i} depth,total {len(files)} depth")
