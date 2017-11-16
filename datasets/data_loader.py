import torch
from torchvision import datasets
from torchvision import transforms


def loadCIFAR10(opt):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    data = {}
    
    data['train_loader'] = torch.utils.data.DataLoader(
        datasets.CIFAR10(root='./data', train=True, transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            normalize,
        ]), download=True),
        batch_size=opt.batch_size, shuffle=True,
        num_workers=opt.workers, pin_memory=True)

    data['val_loader'] = torch.utils.data.DataLoader(
        datasets.CIFAR10(root='./data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=opt.batch_size, shuffle=False,
        num_workers=opt.workers, pin_memory=True)
    return data
