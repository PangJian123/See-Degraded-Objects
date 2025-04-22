import numpy as np
from torch.utils.data import Dataset
import os
import torch
import json
from PIL import Image
from lxml import etree
import copy
from scipy.ndimage import zoom

class VOCDataSet(Dataset):
    """读取解析PASCAL VOC2007/2012数据集"""

    def __init__(self, voc_root, year="2012", transforms=None, txt_name: str = "train.txt", json_name=None, mode=None, prior=None):
        assert year in ["2007", "2012"], "year must be in ['2007', '2012']"
        self.root = voc_root
        self.img_root = os.path.join(self.root, "JPEGImages")
        self.annotations_root = os.path.join(self.root, "Annotations")
        self.mode = mode
        self.prior = prior

        # read train.txt or val.txt file
        txt_path = os.path.join(self.root, "ImageSets", "Main", txt_name)
        assert os.path.exists(txt_path), "not found {} file.".format(txt_path)

        with open(txt_path) as read:
            xml_list = [os.path.join(self.annotations_root, line.strip() + ".xml")
                        for line in read.readlines() if len(line.strip()) > 0]

        self.xml_list = []
        # check file
        for xml_path in xml_list:
            if os.path.exists(xml_path) is False:
                print(f"Warning: not found '{xml_path}', skip this annotation file.")
                continue

            # check for targets
            with open(xml_path) as fid:
                xml_str = fid.read()
            xml = etree.fromstring(xml_str.encode('utf-8'))
            data = self.parse_xml_to_dict(xml)["annotation"]
            if "object" not in data:
                print(f"INFO: no objects in {xml_path}, skip this annotation file.")
                continue

            num_box = 0
            for obj in data['object']:
                if "traffic_sign" in obj["name"] or "traffic_light" in obj["name"]:
                    continue
                else:
                    num_box += 1
            if num_box == 0:
                print(f"INFO: only traffic_sign and traffic_light in {xml_path}, skip this annotation file.")
                continue

            self.xml_list.append(xml_path)

        assert len(self.xml_list) > 0, "in '{}' file does not find any information.".format(txt_path)

        # read class_indict
        json_file = f'./{json_name}'
        assert os.path.exists(json_file), "{} file not exist.".format(json_file)
        with open(json_file, 'r') as f:
            self.class_dict = json.load(f)

        self.transforms = transforms

    def __len__(self):
        return len(self.xml_list)

    def __getitem__(self, idx):
        xml_path = self.xml_list[idx]
        with open(xml_path) as fid:
            xml_str = fid.read()
        xml = etree.fromstring(xml_str.encode('utf-8'))
        data = self.parse_xml_to_dict(xml)["annotation"]
        img_path = os.path.join(self.img_root, data["filename"])
        image = Image.open(img_path)

        if self.prior == 'trans':
            # load transmission map
            trans_map_path = img_path.replace('JPEGImages', 'transmission')
            trans_map = Image.open(trans_map_path)
        elif self.prior == 'ams':
            # # load ams image
            ams_map_path = img_path.replace('JPEGImages', 'ams_img').replace('jpg', 'png')
            ams_map = Image.open(ams_map_path)
        elif self.prior == 'fusion' or self.prior == 'weight_fusion':
            # load transmission map and ams image
            trans_map_path = img_path.replace('JPEGImages', 'transmission')
            trans_map = Image.open(trans_map_path)
            ams_map_path = img_path.replace('JPEGImages', 'ams_img').replace('jpg', 'png')
            ams_map = Image.open(ams_map_path)
        elif self.prior == 'inversedepth':
            inversedepth_path = img_path.replace('JPEGImages', 'inversedepth').replace('jpg', 'png')
            inversedepth = Image.open(inversedepth_path)
        else:
            assert print('no prior in args')

        boxes = []
        labels = []
        iscrowd = []
        assert "object" in data, "{} lack of object information.".format(xml_path)
        for obj in data["object"]:
            if obj["name"] == 'motorcycle': # motorbike in RTTS, motorcycle in FOC,
                obj["name"] = 'motorbike'
            if "traffic_sign" in obj["name"] or "traffic_light" in obj["name"] or obj["name"] not in self.class_dict.keys():  # skip two classes in FOG_CITY
                continue

            xmin = float(obj["bndbox"]["xmin"])
            xmax = float(obj["bndbox"]["xmax"])
            ymin = float(obj["bndbox"]["ymin"])
            ymax = float(obj["bndbox"]["ymax"])

            # 进一步检查数据，有的标注信息中可能有w或h为0的情况，这样的数据会导致计算回归loss为nan
            if xmax <= xmin or ymax <= ymin:
                print("Warning: in '{}' xml, there are some bbox w/h <=0".format(xml_path))
                assert 0
                # continue
            if (xmax - xmin) <= 3 or (ymax - ymin) <= 3:
                # remove the small object, avoiding the cropped object is invalid
                continue

            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(self.class_dict[obj["name"]])
            if "difficult" in obj:
                iscrowd.append(int(obj["difficult"]))
            else:
                iscrowd.append(0)

        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)
        image_id = torch.tensor([idx])

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["iscrowd"] = iscrowd
        target["filename"] = data["filename"]

        # # # draw_box
        # from draw_box_utils import draw_objs
        # from torchvision.transforms import transforms
        # draw_image = image
        # boxes = target['boxes'].cpu()
        # classes = target['labels'].cpu()
        # scores = torch.ones_like(target['labels']).cpu()
        # cate = {
        #     "1":'person',
        #     "2":'car',
        #     "3":'bus',
        #     "4":'bicycle',
        #     "5":'motorbike',
        # }
        # a = draw_objs(image, boxes, classes, scores, category_index=cate)
        # a.save('img_with_box.jpg')
        if self.transforms is not None:
            if self.prior == 'trans':
                image, target, trans_map = self.transforms(image, target, trans_map)
                # transfer the trans_map to depth_map
                negative_depth = torch.log(trans_map + 0.00000000001)
                depth = (-negative_depth) / 0.001
                return image, target, depth
            elif self.prior == 'ams':
                # # -----------------------------------using ams img
                image, target, ams_map = self.transforms(image, target, ams_map)
                return image, target, ams_map
            elif self.prior == 'fusion':
                image, target, ams_map, trans_map = self.transforms(image, target, ams_map, trans_map)
                negative_depth = torch.log(trans_map + 0.00000000001)
                depth = (-negative_depth) / 0.001
                fusion_map = 0.5 * depth + 0.5 * ams_map
                return image, target, fusion_map
            elif self.prior == 'weight_fusion':
                image, target, ams_map, trans_map = self.transforms(image, target, ams_map, trans_map)
                negative_depth = torch.log(trans_map + 0.00000000001)
                depth = (-negative_depth) / 0.001
                if ams_map.shape[1:] != depth.shape[1:]:
                    print(f'ams:{ams_map.shape[1:]}, trans:{depth.shape[1:]}')
                return image, target, [ams_map, depth]
            elif self.prior == 'inversedepth':
                # # load inversedepth
                image, target, inversedepth = self.transforms(image, target, inversedepth)
                return image, target, inversedepth
            else:
                assert print('no prior in args')


    def get_height_and_width(self, idx):
        # read xml
        xml_path = self.xml_list[idx]
        with open(xml_path) as fid:
            xml_str = fid.read()
        xml = etree.fromstring(xml_str.encode('utf-8'))
        data = self.parse_xml_to_dict(xml)["annotation"]
        data_height = int(data["size"]["height"])
        data_width = int(data["size"]["width"])
        return data_height, data_width

    def parse_xml_to_dict(self, xml):
        """
        将xml文件解析成字典形式，参考tensorflow的recursive_parse_xml_to_dict
        Args:
            xml: xml tree obtained by parsing XML file contents using lxml.etree

        Returns:
            Python dictionary holding XML contents.
        """

        if len(xml) == 0:  # 遍历到底层，直接返回tag对应的信息
            return {xml.tag: xml.text}

        result = {}
        for child in xml:
            child_result = self.parse_xml_to_dict(child)  # 递归遍历标签信息
            if child.tag != 'object':
                result[child.tag] = child_result[child.tag]
            else:
                if child.tag not in result:  # 因为object可能有多个，所以需要放入列表里
                    result[child.tag] = []
                result[child.tag].append(child_result[child.tag])
        return {xml.tag: result}

    def coco_index(self, idx):
        """
        该方法是专门为pycocotools统计标签信息准备，不对图像和标签作任何处理
        由于不用去读取图片，可大幅缩减统计时间

        Args:
            idx: 输入需要获取图像的索引
        """
        # read xml
        xml_path = self.xml_list[idx]
        with open(xml_path) as fid:
            xml_str = fid.read()
        xml = etree.fromstring(xml_str.encode('utf-8'))
        data = self.parse_xml_to_dict(xml)["annotation"]
        data_height = int(data["size"]["height"])
        data_width = int(data["size"]["width"])
        # img_path = os.path.join(self.img_root, data["filename"])
        # image = Image.open(img_path)
        # if image.format != "JPEG":
        #     raise ValueError("Image format not JPEG")
        boxes = []
        labels = []
        iscrowd = []

        for obj in data["object"]:
            if obj["name"] == 'motorcycle':
                obj["name"] = 'motorbike'
            if "traffic_sign" in obj["name"] or "traffic_light" in obj["name"] or obj["name"] not in self.class_dict.keys():  # skip two classes in FOG_CITY
                continue
            xmin = float(obj["bndbox"]["xmin"])
            xmax = float(obj["bndbox"]["xmax"])
            ymin = float(obj["bndbox"]["ymin"])
            ymax = float(obj["bndbox"]["ymax"])
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(self.class_dict[obj["name"]])
            iscrowd.append(int(obj["difficult"]))

        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        return (data_height, data_width), target

    @staticmethod
    def collate_fn(batch):
        return tuple(zip(*batch))


