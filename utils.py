import torch
import torchvision.transforms as transforms
import torchvision
from torch.utils.data import DataLoader
import Pruner.vgg_pruner as VGGpruner
import Pruner.resnet_pruner as RESNETpruner
from Models.c_resnet import resnet50
from Models.c_vgg import compress_vgg19_bn
import os
#config
CIFAR_MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
CIFAR_STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)



def cifar_test_dataloader(cifar,mean, std, batch_size=16, num_workers=2, shuffle=True):


    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    if cifar == 10:
        cifar_test_data = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

    if cifar == 100:
        cifar_test_data = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)

    cifar_test_loader = DataLoader(
        cifar_test_data, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

    return cifar_test_loader


def cifar_train_dataloader(cifar,mean, std, batch_size=16, num_workers=2, shuffle=True):


    transform_train = transforms.Compose([
        #transforms.ToPILImage(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    if cifar == 10:
        cifar_train_data = torchvision.datasets.CIFAR10(root='./data', train=True, download=True,
                                                       transform=transform_train)

    if cifar == 100:
        cifar_train_data = torchvision.datasets.CIFAR100(root='./data', train=True, download=True,
                                                        transform=transform_train)
    cifar_training_loader = DataLoader(
        cifar_train_data, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

    return cifar_training_loader

def load_compress_model(path,cifar=100, net="vgg"):
    checkpoint=torch.load(path)
    if "cfg" in checkpoint:
        cfg=checkpoint["cfg"]
    else:
        cfg=None
    if "mask" in checkpoint:
        mask=checkpoint["mask"]
    else:
        mask=None
    if "codebook" in checkpoint:
        codebook=checkpoint["codebook"]
    else:
        codebook=None

    if net == "vgg":
        if not cfg:
            cfg=[64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']

        model = compress_vgg19_bn(cnfg=cfg,num_class=cifar)
        model.load_state_dict(checkpoint["state_dict"])
        if mask:

            model,masks=VGGpruner.prune_weights(model,masks=mask)
            #     pass


    if net == "resnet":

        if not cfg:
            cfg=[[64],3*[64],4*[128],6*[256],3*[512]]

        model = resnet50(cfg=cfg,num_class=cifar)
        model.load_state_dict(checkpoint["state_dict"])
        if mask:

            model,mask=RESNETpruner.prune_weights(model,masks=mask)




    return model,cfg,mask, codebook

def save_compress_model(model,path,cfg=None,mask=None, codebook=None):


    torch.save({ 'state_dict': model.state_dict() ,'cfg': cfg, 'mask':mask, 'codebook':codebook}, path)