

import os
import sys
import argparse
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import time
from torch.utils.data import DataLoader
from utils import cifar_test_dataloader,save_compress_model,load_compress_model
from Models.vgg import vgg19_bn, vgg19_bn_pr
# from dataset import *
from torch.autograd import Variable
import Quantizer.vgg_quantizer as quantizer
import Pruner.vgg_pruner as pruner


#from utils import WarmUpLR

start = time.time()


def train(epoch,net,cifar_training_loader,optimizer):
    net.train()
    for batch_index, (images, labels) in enumerate(cifar_training_loader):
        # if epoch <= args.warm:
        #     warmup_scheduler.step()

        labels = labels.cuda()
        images = images.cuda()

        optimizer.zero_grad()
        outputs = net(images)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()

        n_iter = (epoch - 1) * len(cifar_training_loader) + batch_index + 1



        print('Training Epoch: {epoch} [{trained_samples}/{total_samples}]\tLoss: {:0.4f}\tLR: {:0.6f}'.format(
            loss.item(),
            optimizer.param_groups[0]['lr'],
            epoch=epoch,
            trained_samples=batch_index * args.b + len(images),
            total_samples=len(cifar_training_loader.dataset)
        ))




def eval_training(epoch,net):
    net.eval()

    test_loss = 0.0  # cost function error
    correct = 0.0

    for (images, labels) in cifar_test_loader:
        images = Variable(images)
        labels = Variable(labels)

        images = images.cuda()
        labels = labels.cuda()

        outputs = net(images)
        loss = loss_function(outputs, labels)
        test_loss += loss.item()
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum()

    print('Test set: Average loss: {:.4f}, Accuracy: {:.4f}'.format(
        test_loss / len(cifar_test_loader.dataset),
        correct.float() / len(cifar_test_loader.dataset)
    ))
    print()

    return correct.float() / len(cifar_test_loader.dataset)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # parser.add_argument('-net', type=str, required=True, help='net type')
    # parser.add_argument('-gpu', type=bool, default=True, help='use gpu or not')
    parser.add_argument('-cifar', type=int, default=10, choices=[10, 100],
                        help='number of workers for dataloader')
    parser.add_argument('-w', type=int, default=2, help='number of workers for dataloader')
    parser.add_argument('-b', type=int, default=64, help='batch size for dataloader')
    parser.add_argument('-s', type=bool, default=True, help='whether shuffle the dataset')
    parser.add_argument('-warm', type=int, default=1, help='warm up training phase')
    parser.add_argument('-lr', type=float, default=0.0005, help='initial learning rate')
    args = parser.parse_args()

    # load pruned model

    CIFAR_MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
    CIFAR_STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)


    """TMP"""

    # directory to save weights file
    CHECKPOINT_PATH = 'checkpoint'

    # total training epoches
    EPOCH = 12
    MILESTONES = [5]

    # initial learning rate
    # INIT_LR = 0.1

    # time of we run the script
    TIME_NOW = datetime.now().isoformat()


    # save weights file per SAVE_EPOCH epoch
    SAVE_EPOCH = 5
    NAMES=["b_vgg1","b_vgg2","b_vgg3","b_vgg4"]

    """"""




    model1, cfg1 ,mask1,codebook1= load_compress_model(
        "quant", cifar=args.cifar)
    model1.cuda()
    model2, cfg2,mask2,codebook2 = load_compress_model(
        "quant", cifar=args.cifar)
    model2.cuda()
    model3, cfg3, mask3, codebook3 = load_compress_model(
        "quant", cifar=args.cifar)
    model3.cuda()
    model4, cfg4, mask4, codebook4 = load_compress_model(
        "quant", cifar=args.cifar)
    model4.cuda()

    models = [model1,model2,model3,model4]
    print(models[1])

    cfgs=[cfg1,cfg2,cfg3,cfg4]
    masks=[mask1,mask2,mask3,mask4]
    codebooks=[codebook1,codebook2,codebook3,codebook4]


    optimizers=[]
    train_shedulers=[]
    for i in range(4):

        optimizer = optim.SGD(models[i].parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
        optimizers.append(optimizer)
        train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[4,7,11,12],
                                                         gamma=0.5)  # learning rate decay
        train_shedulers.append(train_scheduler)
    #net.load_state_dict(torch.load("checkpoint/vgg/2019-08-05T14:16:37.621728/vgg-26-best.pth"), args.gpu)



    #load 3 datasets
    # data preprocessing:
    train_loaders=[]

    transform_train = transforms.Compose([
        # transforms.ToPILImage(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD)
    ])


    cifar_train_data = torchvision.datasets.CIFAR10(root='./data', train=True, download=True,
                                                      transform=transform_train)

    samples1 = torch.utils.data.RandomSampler(cifar_train_data, replacement=True, num_samples=50000)
    samples2 = torch.utils.data.RandomSampler(cifar_train_data, replacement=True, num_samples=50000)
    samples3 = torch.utils.data.RandomSampler(cifar_train_data, replacement=True, num_samples=50000)
    samples4 = torch.utils.data.RandomSampler(cifar_train_data, replacement=True, num_samples=50000)
    """
    i1 = []
    all=np.arange(50000)
    for s in samples1:
        i1.append(s)
    unique1, counts = np.unique(i1, return_counts=True)
    samples2 = [item for item in all if item not in unique1]
    print(len(unique1))
    print(len(samples2))

    need = 50000 - len(samples2)

    samples_tmp = torch.utils.data.RandomSampler(cifar_train_data, replacement=True, num_samples=need)

    for z in samples_tmp:
        samples2.append(z)

    print(len(samples2))

    samples=[samples1,samples2]
    """
    samples = [samples1, samples2,samples3,samples4]
    for i in range(4):


        bs = torch.utils.data.BatchSampler(samples[i], batch_size=64, drop_last=False)
        train_loader = torch.utils.data.DataLoader(cifar_train_data, batch_sampler=bs, num_workers=2)
        train_loaders.append(train_loader)


    cifar_test_loader = cifar_test_dataloader(
        args.cifar,
        CIFAR_MEAN,
        CIFAR_STD,
        num_workers=args.w,
        batch_size=args.b,
        shuffle=args.s
    )

    loss_function = nn.CrossEntropyLoss()


    iter_per_epoch = len(train_loaders[0])
    #warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * args.warm)
    checkpoint_path = os.path.join(CHECKPOINT_PATH, "bag_vgg", TIME_NOW)



    # create checkpoint folder to save model
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    checkpoint_path = os.path.join(checkpoint_path, '{net}-{epoch}-{type}.pth')

    best_accs = [0.0, 0.0 ,0.0,0.0]
    for epoch in range(1,15):
        for i in range (4):
            print("model"+str(i)+ " training")
            # if epoch > args.warm:
            #     train_scheduler.step(epoch)
            train_shedulers[i].step(epoch)
            train(epoch,models[i],train_loaders[i],optimizers[i])
            models[i],masks[i]=pruner.prune_weights(models[i],masks=masks[i])
            models[i]=quantizer.fixed_point_quant(models[i])
            models[i].cuda()
            acc = eval_training(epoch,models[i])

            # start to save best performance model after learning rate decay to 0.01
            if epoch > 0 and best_accs[i] < acc:
                #torch.save(net.state_dict(), checkpoint_path.format(net=NAME, epoch=epoch, type='best'))
                save_compress_model(models[i], checkpoint_path.format(net=NAMES[i], epoch=epoch, type='best'), cfg=cfg1)
                best_accs[i] = acc
                continue

            if not epoch % 6:
                save_compress_model(models[i],checkpoint_path.format(net=NAMES[i], epoch=epoch, type='regular'),cfg=cfg1)
                #torch.save(net.state_dict(), checkpoint_path.format(net=NAME, epoch=epoch, type='regular'))

    print(time.time() - start)

