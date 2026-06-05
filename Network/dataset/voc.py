from torchvision.datasets import VOCSegmentation
import torch
import torchvision
from torch.utils.data import DataLoader
import numpy as np
import os
from PIL import Image


def VOCtransforms(img, target):
    ToTensor = torchvision.transforms.ToTensor()
    return ToTensor(img), ToTensor(target)


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

train_size = 512
train_transforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize([train_size, train_size]),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
test_transforms = torchvision.transforms.Compose([torchvision.transforms.Resize([train_size, train_size])])


def transforms(img, target):
    return train_transforms(img), test_transforms(target)


def train_transform(img, target):
    if np.random.randn() > 0.5:
        img = torchvision.transforms.functional.vflip(img)
        target = torchvision.transforms.functional.vflip(target)
    if np.random.randn() > 0.5:
        img = torchvision.transforms.functional.hflip(img)
        target = torchvision.transforms.functional.hflip(target)
    return img, target


class VOCSeg(VOCSegmentation):
    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert("RGB")
        target = Image.open(self.masks[index])
        img = np.array(img)
        target = np.array(target)
        target[target == 255] = 0

        if self.transforms is not None:
            aug = self.transforms(image=img, mask=target)
            img, target = aug['image'], aug['mask'] # use albumentations

        # if self.image_set == 'train':
        #     img, target = train_transform(img, target)
        img = torch.from_numpy(img) / 255
        img = img.permute(2, 0, 1)
        img = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img)

        # img = np.array(img)

        return img, torch.from_numpy(target)


if __name__ == '__main__':

    train_dataset = VOCSeg(root='E:/fuck/Dataset-Tool-Segmentation/data/VOC', year='2012', image_set='train',
                           download=False, transforms=transforms)
    train_loader = DataLoader(dataset=train_dataset, batch_size=16, num_workers=2, shuffle=False)
    print(len(train_loader))
    for i, j in train_dataset:
        i = np.array(i)
        j = np.array(j)

    '''
    mean: [0.44610919 0.42568388 0.39048806]
    std: [0.23871837 0.23377924 0.23621951]

    imgdir = './data/VOCdevkit/VOC2007/JPEGImages'
    file = open('./data/VOCdevkit/VOC2007/ImageSets/Segmentation/train.txt', 'r')
    lines = file.readlines()
    cum_mean = np.zeros(3)
    cum_std = np.zeros(3)
    for i in lines:
        if len(i) != 7:
            break
        i = i[:-1] + '.jpg'
        imgpath = os.path.join(imgdir, i)
        img = np.array(Image.open(imgpath)) / 255.
        zz = img.mean(axis=(0, 1))
        img = img.reshape(-1, 3)
        z = img.mean(axis=0)
        cum_mean += img.mean(axis=0)
        cum_std += img.std(axis=0)
    mean = cum_mean / len(lines)
    std = cum_std / len(lines)
    print(f"mean: {mean}")
    print(f"std: {std}")
    '''

