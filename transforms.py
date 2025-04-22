import random
from torchvision.transforms import functional as F


class Compose(object):
    """组合多个transform函数"""
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target, clear_image=None, extra_image=None):
        if clear_image is None:
            for t in self.transforms:
                image, target = t(image, target)
            return image, target
        else:
            if extra_image is None:
                for t in self.transforms:
                    image, target, clear_image = t(image, target, clear_image)
                return image, target, clear_image
            else:
                for t in self.transforms:
                    image, target, clear_image, extra_image = t(image, target, clear_image, extra_image)
                return image, target, clear_image, extra_image

class ToTensor(object):
    """将PIL图像转为Tensor"""
    def __call__(self, image, target, clear_image=None, extra_prior=None):
        if clear_image is None:
            image = F.to_tensor(image)
            return image, target
        else:
            if extra_prior is None:
                image = F.to_tensor(image)
                clear_image = F.to_tensor(clear_image)
                return image, target, clear_image
            else:
                image = F.to_tensor(image)
                clear_image = F.to_tensor(clear_image)
                extra_image = F.to_tensor(extra_prior)
                return image, target, clear_image, extra_image


class RandomHorizontalFlip(object):
    """随机水平翻转图像以及bboxes"""
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, target, clear_image=None, extra_prior=None):
        if clear_image is None:
            if random.random() < self.prob:
                height, width = image.shape[-2:]
                image = image.flip(-1)  # 水平翻转图片
                bbox = target["boxes"]
                # bbox: xmin, ymin, xmax, ymax
                bbox[:, [0, 2]] = width - bbox[:, [2, 0]]  # 翻转对应bbox坐标信息
                target["boxes"] = bbox
            return image, target
        else:
            if extra_prior is None:
                if random.random() < self.prob:
                    height, width = image.shape[-2:]
                    image = image.flip(-1)  # 水平翻转图片
                    bbox = target["boxes"]
                    # bbox: xmin, ymin, xmax, ymax
                    bbox[:, [0, 2]] = width - bbox[:, [2, 0]]  # 翻转对应bbox坐标信息
                    target["boxes"] = bbox
                    clear_image = clear_image.flip(-1)
                return image, target, clear_image
            else:
                if random.random() < self.prob:
                    height, width = image.shape[-2:]
                    image = image.flip(-1)  # 水平翻转图片
                    bbox = target["boxes"]
                    # bbox: xmin, ymin, xmax, ymax
                    bbox[:, [0, 2]] = width - bbox[:, [2, 0]]  # 翻转对应bbox坐标信息
                    target["boxes"] = bbox
                    clear_image = clear_image.flip(-1)
                    extra_prior = extra_prior.flip(-1)
                return image, target, clear_image, extra_prior
