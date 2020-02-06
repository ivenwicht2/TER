from torch.utils.data import DataLoader,Dataset,ConcatDataset
import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder

class DatasetFromSubset(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.subset)

def import_img(path) :
    print('importation images : ',end='')
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor()
    ])
    
    datasets = ImageFolder(root=path,transform=transform)
    split_size = int(0.7 * len(datasets))

    train_dataset, test_dataset = torch.utils.data.random_split(datasets, [split_size, len(datasets)-split_size])

    train_set = data_augmentation(train_dataset)
    train_loader = DataLoader(train_set,
                            batch_size=25,
                            shuffle=True
                            )
    test_set = data_augmentation(test_dataset)
    test_loader = DataLoader(test_set,
                            batch_size=25,
                            shuffle=True
                            )
    print('done')
    print('Image number  : ',len(datasets))
    print('New image number : ',(len(train_set)+len(test_set)))

    return train_loader, test_loader

def data_augmentation(dataset):

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])

    d1 = DatasetFromSubset(dataset,transform)

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomResizedCrop(224),
        transforms.ToTensor()
    ])


    d2 =  DatasetFromSubset(dataset,transform)

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ])

    d3 =  DatasetFromSubset(dataset,transform)

    return ConcatDataset([d1,d2,d3])


