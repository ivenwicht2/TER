from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
import torch 
from model_build import  detection
from dataset import PennFudanDataset
import transforms as T
import numpy as np
import torchvision 
from torchvision import transforms
from torch import nn,optim
import utils
import math 
from engine import *

def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


def train():

    dataset = PennFudanDataset('VOCdevkit',get_transform(train=True))
    dataset_test = PennFudanDataset('VOCdevkit',get_transform(train=False))
    indices = torch.randperm(len(dataset)).tolist()
    dataset = torch.utils.data.Subset(dataset, indices[:-50])
    dataset_test = torch.utils.data.Subset(dataset_test, indices[-50:])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=3, shuffle=True, num_workers=0,
        collate_fn=utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=2, shuffle=True, num_workers=0,collate_fn=utils.collate_fn)
    print("data loader : done ")
    
    
    model = detection(10)
    model.to(device)
    print("model importation : done ")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    


    for epoch in range(50):
        
        print("epoch {} : ".format(epoch),end="")
        for batch ,(images, targets) in enumerate(data_loader) : 
           print("*",end="")
           model.train()
           model.zero_grad()
           targets = [x for x in targets]
           loss_dict = model(images, targets)
           losses = sum(loss for loss in loss_dict.values())
           loss_dict_reduced = utils.reduce_dict(loss_dict)
           losses_reduced = sum(loss for loss in loss_dict_reduced.values())
           loss_value = losses_reduced.item()
           if not math.isfinite(loss_value):
               print("Loss is {}, stopping training".format(loss_value))
               print(loss_dict_reduced)
               sys.exit(1)           
           optimizer.zero_grad()
           losses.backward()
           optimizer.step()
        
        exp_lr_scheduler.step()
        print()
        evaluate(model,data_loader_test,device=device)

if __name__ == "__main__":
    train()

