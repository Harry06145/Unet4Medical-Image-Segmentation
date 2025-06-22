# 使用更标准化的预处理
from torchvision import transforms
import torch
import numpy as np
# 创建数据时应用标准化
class bulid_Dataset(torch.utils.data.Dataset):
    def __init__(self, images, masks,mean,std):
        self.images = images
        self.masks = masks
        self.mean = mean
        self.std = std

        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        
        img = (self.images[index]-self.mean) / self.std
        mask = self.masks[index]
        
        return torch.tensor(img, dtype=torch.float32), torch.tensor(mask, dtype=torch.float32)