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

class ImgProcessor:
    model = None
    v_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    def __init__(self):
        self.model = seg.fcn_resnet101(pretrained=True, progress=True)
        self.model.cuda()
        self.model.eval()
    
    def process(self, imgPath):
        #Load image
        img_npp = Image.open(imgPath).convert('RGB')
        img = self.v_transform(img_npp)
        img = img.unsqueeze(0)
        img = Variable(img.cuda())

        #Get result
        res = self.model(img)
        _, tmp = res.popitem()[1].squeeze(0).max(0)
        res_seg = tmp.data.cpu().numpy().squeeze()
        conv = Image.fromarray(res_seg.astype('uint8'), 'RGB')

        return conv.convert('L')

p = ImgProcessor()
res = p.process('skin-guess/cropped.png')
plt.imshow(img_not_preprocessed)
plt.show()