import csv
import json
import logging
import os
import re
import difflib
import sys
import cv2
import torch
import random
from abc import abstractmethod
from itertools import islice
from scipy import ndimage
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
from collections.abc import Mapping
from torch.utils.data import DataLoader
import PIL
import SimpleITK as sitk
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from tqdm import tqdm
from torchvision import transforms
from collections import defaultdict
from PIL import Image
import math
from monai.transforms import RandSpatialCrop

# CONDITIONS = [
#     'enlarged cardiomediastinum',
#     'cardiomegaly',
#     'lung opacity',
#     'lung lesion',
#     'edema',
#     'consolidation',
#     'pneumonia',
#     'atelectasis',
#     'pneumothorax',
#     'pleural effusion',
#     'pleural other',
#     'fracture',
#     'support devices',
#     'no finding',
# ]

CONDITIONS = [
    '纵隔心影增大',
    '心脏肥大',
    '肺部阴影',
    '肺部病变',
    '水肿',
    '实变',
    '肺炎',
    '肺不张',
    '气胸',
    '胸腔积液',
    '胸膜其他病变',
    '骨折',
    '支撑装置',
    '未见异常',
]

SCORES = [
    '<BLA>',
    '<POS>',
    '<NEG>',
    '<UNC>'
]

Token_to_Text = {
    '<BLA>': 'blank',
    '<POS>': 'positive',
    '<NEG>': 'negative',
    '<UNC>': 'uncertain'
}


class CTRG_Dataset(Dataset):
    """_summary_
    Args:
        Dataset (_type_): _description_: modality asked task formulated as vqa task for Radiopaedia dataset
        csv_path (_type_): path to csv file
    Output:
        Dict: {
             "image_dict": [{"image": image, "position": {"question": 0}}], # image is a tensor of shape [c,w,h,d] like, [3,512,512,4], position is a dict, random choice of 0 or len(question)
            "question": question, # random choice of caption prompts
            "answer":answer, # caption
            }
    """

    def __init__(self, data_dir, data_json, label_path, split):
        self.data_dir = data_dir
        self.split = split

        with open(data_json, 'r') as f:
            data_dict = json.load(f)
        self.data_list = data_dict[split]

        self.label = self._load_label(label_path)

    def _load_label(self, label_path):
        label_dict = {}

        data = pd.read_csv(label_path)
        for index, row in data.iterrows():
            idx = row['id']
            label = row[1:].to_list()

            label_dict[idx] = label

        return label_dict

    def resize_image(self, image):
        if len(image.shape) == 3:
            if image.shape[0] < image.shape[2]:
                image = image.transpose(1, 2, 0)
            # print('before resize',image.shape)
            image = cv2.resize(image, (512, 512),
                               interpolation=cv2.INTER_LINEAR)
            # print('after resize',image.shape)
            image = image[np.newaxis, :, :, :]
            image = np.concatenate([image, image, image], axis=0)
        else:
            print('image shape', image.shape)

        if image.shape[-1] > 64:
            image = ndimage.zoom(
                image, (3/image.shape[0], 512/image.shape[1], 512/image.shape[2], 64/image.shape[3]), order=0)
        else:
            print('image shape', image.shape)
            image = ndimage.zoom(
                image, (3/image.shape[0], 512/image.shape[1], 512/image.shape[2], 64/image.shape[3]), order=0)

        return image

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        image_id = self.data_list[index]['id']
        cls_labels = torch.tensor(
            self.label[int(image_id)], dtype=torch.float32).long()
        img_path = os.path.normpath(os.path.join(
            self.data_dir, str(image_id)+'.nii.gz'))

        itk_image = sitk.ReadImage(img_path)
        image = sitk.GetArrayFromImage(itk_image)
        image = self.resize_image(image)

        image = np.array(image)

        image = (image-image.min())/(image.max()-image.min())
        image = torch.from_numpy(image).float()

        answer = self.data_list[index]['finding']
        cls_labels = torch.where(cls_labels == 1, cls_labels, torch.tensor(0))

        prompt = ""
        for i, l in enumerate(cls_labels):
            disease = CONDITIONS[i]
            if l == 1:
                state = '存在'
                prompt += f"\"{disease}\" ： {state}。 "
            else:
                state = '不存在'
                prompt += f"\"{disease}\" ： {state}。 "

        # guide = "请结合以上CT影像以及给出的关键病症信息，生成一份完整的中文医学影像诊断报告。报告需要详细描述影像所见的解剖结构，并指出任何存在的异常情况。 "
        # 上述复杂的guide效果不好，换用下面的新的更简单的guide
        guide = "影像所见："
        question = prompt + guide

        image_dict = {
            "image": image,
            "position": {
                "question": 0
            }
        }

        return {
            "image_id": image_id,
            "image_dict": [image_dict],
            "guide": guide,
            "question": question,
            "answer": answer,
            "cls_labels": cls_labels,
        }
