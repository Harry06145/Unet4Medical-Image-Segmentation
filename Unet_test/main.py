import torch
from torch import nn
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor, Compose
from torch.utils.data import DataLoader
import torch.optim as optim 
from model import U_net
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
from data_loader import bulid_Dataset
from loss_fn import DcieLoss
from torchvision import transforms

def calculate_mean_std(images):
    mean = np.mean(images)
    std = np.std(images)
    return mean, std

device = "cuda"
batch_size = 2
num_class = 1

image = sitk.ReadImage("/home/harrylee/medical_dataset/ct_scans/VESSEL12_01.mhd") 
image_array = sitk.GetArrayFromImage(image).astype(np.float32)
mean, std = calculate_mean_std(image_array)

mask = sitk.ReadImage("/home/harrylee/medical_dataset/masks/VESSEL12_01.mhd")

mask_array = sitk.GetArrayFromImage(mask).astype(np.float32)

dataset = bulid_Dataset(image_array, mask_array,mean,std)

dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

model = U_net(in_channels=1,out_channels=num_class).to(device)

optimizer = optim.Adam(model.parameters(),lr=0.00001)
loss_fn = DcieLoss()

epochs = 100

for epoch in range(epochs):
    epoch_loss = 0.0
    for idx,(image,label) in enumerate(dataloader):
        imgs = torch.tensor(image).to(device).unsqueeze(1)
        masks = torch.tensor(label).to(device).unsqueeze(1)
        optimizer.zero_grad()
        output = model(imgs)
        loss = loss_fn(output,masks)
        loss.backward()
        optimizer.step()
        epoch_loss += loss

        if idx == 177:
            np.save("./otuput.npy",output[0].detach().cpu().numpy())
            np.save("./labels.npy",masks[0].detach().cpu().numpy())
            np.save("./images.npy",imgs[0].detach().cpu().numpy())

    print(f"epoch{epoch+1} loss:", epoch_loss.item()/idx)


print("Done")