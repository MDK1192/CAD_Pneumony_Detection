# -*- coding: utf-8 -*-
"""
Created on Sat Dec  5 11:05:46 2020

@author: Marc
"""

# Imports here
import matplotlib.pyplot as plt
import pandas as pd
import torch
import json
from torch import nn
from torch import optim
import numpy as np
from PIL import Image
from torchvision import datasets, transforms, models
from collections import OrderedDict
from os import walk
 
  
def check_images(path):    



        
    #load model
    model = torch.load('model_google_trained_val4.pth')
    classification = []
    percentages_0 = []
    percentages_1 = []
    files = []
    path_normal = path + '/Test/Normal'
    path_anomaly = path + '/Test/Anomaly'
    for (dirpath, dirnames, filenames) in walk(path_anomaly):
        for file in filenames:
            files.extend([str(file)])
        break
    for (dirpath, dirnames, filenames) in walk(path_normal):
        for file in filenames:
            files.extend([str(file)])
        break

        
    device = "cpu"
    data_transforms = transforms.Compose([transforms.Resize(255),
                                transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406],
                                                    [0.229, 0.224, 0.225])])
    test_data = datasets.ImageFolder(path, transform =data_transforms)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=1000)
    model.eval()
    images, labels = next(iter(testloader))
    # Move input and label tensors to the default device
    images, labels, model = images.to(device), labels.to(device), model.to(device)
    # Get the class probabilities
    ps = torch.exp(model(images))
    top_p, top_class = ps.topk(1, dim=1)
    equals = top_class == labels.view(*top_class.shape)
    for i in range(0,len(top_class.tolist())):
      classification.extend([str(top_class.tolist()[i][0])])
      percentages_0.extend([str(ps.tolist()[i][0])])
      percentages_1.extend([str(ps.tolist()[i][1])])
    data = {'files': files, 'classification': classification, 'percentages_0': percentages_0, 'percentages_1': percentages_1}
    print(data)
    df_results = pd.DataFrame(data=data)
    return df_results

    


