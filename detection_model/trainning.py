from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
import torch 
from model_build import  detection
from monoset import import_dataset
from dataset import PennFudanDataset
import transforms as T
import utils
import numpy as np
import torchvision 
from torchvision import transforms
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from torch import nn,optim

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
        dataset, batch_size=2, shuffle=True, num_workers=0,
        collate_fn=utils.collate_fn,pin_memory=False)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=4,
        collate_fn=utils.collate_fn)
    print("data loader : done ")
    #model = detection(10)
    #model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)    
    

    backbone = torchvision.models.mobilenet_v2(pretrained=True).features
    backbone.out_channels = 1280
    anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
                                   aspect_ratios=((0.5, 1.0, 2.0),))
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=[0],
                                                output_size=7,
                                                sampling_ratio=2)

    model = FasterRCNN(backbone,
                   num_classes=2,
                   rpn_anchor_generator=anchor_generator,
                   box_roi_pool=roi_pooler)

    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer_ft = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
 
    for epoch ,(images, targets) in enumerate(data_loader) : 
       print("epoch : " , epoch)
       targets = [x for x in targets]
       print(type(images))
       #targets.to(device),images.to(device)
       model(images, targets)

if __name__ == "__main__":
    train()

