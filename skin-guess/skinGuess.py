#%%
#from os import listdir
#DATASET = "D:\DataSets"
#print(listdir(DATASET))

#based off of
#https://github.com/warmspringwinds/pytorch-segmentation-detection/blob/master/pytorch_segmentation_detection/recipes/pascal_voc/segmentation/resnet_34_8s_demo.ipynb
import sys, os
from PIL import Image
from matplotlib import pyplot as plt

import torch
from torchvision import transforms
from torch.autograd import Variable
from torchvision.models import segmentation as seg

import numpy as np

#from torchvision.models

#load image
img_path = 'skin-guess/cropped.png'
valid_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

img_npp = Image.open(img_path).convert('RGB')
img = valid_transform(img_npp)
img = img.unsqueeze(0)
img = Variable(img.cuda())

#%%
#load model
model = seg.fcn_resnet101(pretrained=True, progress=True)
model.cuda()
model.eval()

#%%
#result
res = model(img)
_, tmp = res.squeeze(0).max(0)
res_seg = tmp.data.cpu().numpy().squeeze()

#%% Original Image
plt.imshow(img_npp)
plt.show()

#%%
plt.imshow(res_seg)
plt.show()