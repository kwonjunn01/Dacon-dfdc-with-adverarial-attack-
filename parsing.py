import argparse
import json
import os
from os import cpu_count
from typing import Type
from facenet_pytorch import MTCNN

from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
import torch
import torch.utils.data as data
from PIL import Image
import torchvision.transforms as transforms
import easydict
import cv2

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("device:", device)


args = easydict.EasyDict({
    "root": "/home/diml/ddrive/dataset/deepfake_1st",
    "train_list": "train_list_1st.txt",
    "test_list": "test_list_1st.txt"
})


class ImageRecord(object):
    def __init__(self, row):
        self._data = row

    @property
    def path(self):
        return self._data[0]

    @property
    def label(self):
        return int(self._data[1])


class DFDCDatatset(data.Dataset):
    def __init__(self, root_path, list_file, transform=None):
        self.root_path = root_path
        self.list_file = list_file
        self.transform = transform

        self._parse_list()

    def _load_image(self, image_path):
        return Image.open(image_path).convert('RGB')

    def _parse_list(self):
        self.image_list = [ImageRecord(x.strip().split(' ')) for x in open(self.list_file)]

    def __getitem__(self, index):
        record = self.image_list[index]
        temp = os.path.join(args.root, record.path)
        image = self._load_image(temp)

        if self.transform is not None:
            image = self.transform(image)

        return image, record.label

    def __len__(self):
        return len(self.image_list)


mtcnn = MTCNN(select_largest=False, device=device)

print(args.train_list)

dataset = DFDCDatatset(args.root, args.train_list,transform=transforms.ToTensor())

loader = DataLoader(dataset, shuffle=False, batch_size=2 )


for i, (image, label) in enumerate(loader):
    image =image
    img_cropped = mtcnn(image.transpose(2,3).transpose(1,2))
    cv2.imwrite(os.path.join(args.root, "{}.png".format(i), img_cropped))