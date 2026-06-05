import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np

class UDDPatchDataset(Dataset):
    def __init__(self, txt_file, transforms=None):
        self.samples = []
        with open(txt_file, "r") as f:
            for line in f:
                img_path, mask_path = line.strip().split()
                self.samples.append((img_path, mask_path))
        self.transforms = transforms

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path = self.samples[idx]

        # 读取图片和 mask
        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)  # 这里 mask 已经是 class id，单通道图像
        img = np.array(img)
        mask = np.array(mask)

        if self.transforms is not None:
            aug = self.transforms(image=img, mask=mask)
            img, mask = aug['image'], aug['mask'] # use albumentations

        img = torch.from_numpy(img) / 255
        img = img.permute(2, 0, 1)
        img = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img)

        return img, torch.from_numpy(mask)
