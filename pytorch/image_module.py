from torch.utils.data import DataLoader,Dataset,ConcatDataset
import torch
from torchvision import transforms , datasets
from torchvision.datasets import ImageFolder
from torch.utils.data.sampler import SubsetRandomSampler

import numpy as np  

def import_img(PATH):
    train_transforms = transforms.Compose([transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
                                        transforms.RandomRotation(degrees=15),
                                        transforms.ColorJitter(),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.CenterCrop(size=224),  # Image net standards
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])
                                        ])

    validation_transforms = transforms.Compose([transforms.Resize(256),
                                            transforms.CenterCrop(224),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.485, 0.456, 0.406], 
                                                                 [0.229, 0.224, 0.225])])

    train_data = datasets.ImageFolder(PATH,transform=train_transforms)
    num_workers = 2
    valid_size = 0.3


    num_train = len(train_data)
    indices = list(range(num_train))
    np.random.shuffle(indices)
    valid_split = int(np.floor((valid_size) * num_train))
    valid_idx, train_idx = indices[:valid_split], indices[valid_split:]

    print(len(valid_idx), len(train_idx))

    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=32,
        sampler=train_sampler, num_workers=num_workers)
    valid_loader = torch.utils.data.DataLoader(train_data, batch_size=32, 
        sampler=valid_sampler, num_workers=num_workers)


    return train_loader,valid_loader