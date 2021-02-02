import argparse
import json
import os
from facenet_pytorch import MTCNN
import numpy as np

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
    "train_list": "train_list_fake_2.txt",
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


mtcnn = MTCNN(
    image_size=160, margin=0, min_face_size=20,
    thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
    device=device
)

dataset = DFDCDatatset(args.root, args.train_list, transform=transforms.ToTensor())
# dataset = DFDCDatatset(args.root, args.test_list, transform=transforms.ToTensor())
loader = DataLoader(dataset, shuffle=False, batch_size=1)
dst_path = "/home/diml/ddrive/dataset/dfdc"

for i, (image, label) in tqdm(enumerate(loader)):
    trans = transforms.ToPILImage()
    if label == int(1):
        fake_path = os.path.join(dst_path, 'fake/images')
        cropped_im = mtcnn(trans(image[0]), save_path=os.path.join(fake_path, "img_{:07d}.jpg".format(i+275175+105699+208750)))
        if cropped_im is not None:
            _, __ ,landmarks = mtcnn.detect(trans(cropped_im), landmarks=True)
            flandmark_path = os.path.join(dst_path, "fake/landmarks")
            if landmarks is not None:
                landmarks = np.around(landmarks[0]).astype(np.int16)
                np.save(os.path.join(flandmark_path, "img_{:07d}".format(i+275175+105699+208750)), landmarks)
    # else:
    #     real_path = os.path.join(dst_path, 'real/images')
    #     cropped_im = mtcnn(trans(image[0]), save_path=os.path.join(real_path, "img_{:07d}.jpg".format(i)))
    #     _, __ ,landmarks = mtcnn.detect(trans(cropped_im), landmarks=True)
    #     rlandmark_path = os.path.join(dst_path, "real/landmarks")
    #     if landmarks is not None:
    #         landmarks = np.around(landmarks[0]).astype(np.int16)
    #         np.save(os.path.join(rlandmark_path, "img_{:07d}".format(i)), landmarks)
        
        
    
    