

import os
import argparse
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
import time
from utils import cifar_test_dataloader,cifar_train_dataloader,save_compress_model,load_compress_model

from torch.autograd import Variable
import Quantizer.resnet_quantizer as quantizer
import Pruner.resnet_pruner as RESNETpruner
from Models.c_resnet import resnet50
start = time.time()


def train(epoch):
    net.train()
    for batch_index, (images, labels) in enumerate(cifar_training_loader):
        images = Variable(images)
        labels = Variable(labels)

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




def eval_training(model):
    net=model
    net.eval()
    net.cuda()
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
    parser.add_argument('-cifar', type=int, default=100, choices=[10, 100],
                        help='number of workers for dataloader')
    parser.add_argument('-w', type=int, default=2, help='number of workers for dataloader')
    parser.add_argument('-b', type=int, default=50, help='batch size for dataloader')
    parser.add_argument('-gpu', type=bool, default=True, help='use gpu or not')
    parser.add_argument('-s', type=bool, default=True, help='whether shuffle the dataset')
    parser.add_argument('-lr', type=float, default=0.0005, help='initial learning rate') #0,01 dla poprzedniej
    args = parser.parse_args()

    # load pruned model

    CIFAR_MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
    CIFAR_STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)


    """TMP"""

    # directory to save weights file
    CHECKPOINT_PATH = 'checkpoint'



    # time of we run the script
    TIME_NOW = datetime.now().isoformat()


    # save weights file per SAVE_EPOCH epoch
    SAVE_EPOCH = 0

    NAME="saved_model"
    """"""



    net, cfg, masks, codebooks = load_compress_model("qq", cifar=args.cifar, net="resnet")


    print(cfg)

    net.cuda()
    # data preprocessing:
    cifar_training_loader = cifar_train_dataloader(
        args.cifar,
        CIFAR_MEAN,
        CIFAR_STD,
        num_workers=args.w,
        batch_size=args.b,
        shuffle=args.s
    )

    cifar_test_loader = cifar_test_dataloader(
        args.cifar,
        CIFAR_MEAN,
        CIFAR_STD,
        num_workers=args.w,
        batch_size=args.b,
        shuffle=args.s
    )

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[4,7,11,12], #[6,15,22,29]
                                                     gamma=0.5)  # learning rate decay
    iter_per_epoch = len(cifar_training_loader)
    checkpoint_path = os.path.join(CHECKPOINT_PATH, NAME, TIME_NOW)



    # create checkpoint folder to save model
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    checkpoint_path = os.path.join(checkpoint_path, '{net}-{epoch}-{type}.pth')

    best_acc = 0.0
    for epoch in range(1,14):

        train_scheduler.step(epoch)
        train(epoch)
        cmodel = resnet50(cfg= cfg,num_class=args.cifar)
        cmodel.load_state_dict(net.state_dict())
        #cmodel, masks = RESNETpruner.prune_weights(cmodel, masks=masks)
        cmodel,codebooks = quantizer.kmeans_quant(cmodel,codebooks)

        #net,masks=RESNETpruner.prune_weights(net,masks=masks)
        #
        acc = eval_training(cmodel) #net
        net=cmodel #net
        net.cuda()
        del cmodel

        if epoch > -1 and best_acc < acc:
            #torch.save(net.state_dict(), checkpoint_path.format(net=NAME, epoch=epoch, type='best'))
            save_compress_model(net, checkpoint_path.format(net=NAME, epoch=epoch, type='best'), cfg=cfg,mask=masks,codebook=codebooks)
            best_acc = acc
            continue

        if not epoch % 6:
            save_compress_model(net,checkpoint_path.format(net=NAME, epoch=epoch, type='regular'),cfg=cfg,mask=masks,codebook=codebooks)
            #torch.save(net.state_dict(), checkpoint_path.format(net=NAME, epoch=epoch, type='regular'))

    print(time.time() - start)

