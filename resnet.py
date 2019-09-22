


import argparse
#from dataset import *
import os
#from skimage import io
from matplotlib import pyplot as plt
import torch
import Pruner.resnet_pruner as pruner
import Quantizer.resnet_quantizer as quantizer
from utils import cifar_test_dataloader,load_compress_model,save_compress_model

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-cifar', type=int, default=100, choices=[10,100],
                      help='number of workers for dataloader')
    parser.add_argument('-prune', type=str, default=None, help='prune network')
    parser.add_argument('-quantize', type=str, default=None, help='quantize network')
    parser.add_argument('-gpu', type=bool, default=True, help='use gpu or not')
    parser.add_argument('-w', type=int, default=2, help='number of workers for dataloader')
    parser.add_argument('-b', type=int, default=60, help='batch size for dataloader')
    parser.add_argument('-s', type=bool, default=True, help='whether shuffle the dataset')
    args = parser.parse_args()



    CIFAR_MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
    CIFAR_STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)

    cfg = None
    masks = None
    codebooks = None
    # from Models.c_resnet import resnet50
    # model = resnet50(cfg=[[64],3*[64],4*[128],6*[256],3*[512]],num_class=args.cifar )
    # model.load_state_dict((torch.load("Saved_models/cifar100/resnet50-198-best.pth")))

    net,cfg,masks,codebooks=load_compress_model("Saved_models/orginalModels/cifar100/resnet.pth",cifar=args.cifar,net="resnet")
    model = net



    if args.prune:

        #new_cfg=[[16],3*[64],4*[128],6*[256],3*[256]]
        #model,cfg = pruner.prune_filters(model,new_cfg,args.cifar)

        model,masks = pruner.prune_weights(model, percentage=[90,90,90, 80,90,90,90, 70,90,90,90,80,80, 70,70,70, 90])#  [90,90,90, 80,95,95,90, 80,90,90,95,95,95, 90,95,95, 90])


    if args.quantize:
        model = quantizer.fixed_point_quant(model,bits=8)
        #model,codebooks=quantizer.kmeans_quant(model,k=16)

    cifar_test_loader = cifar_test_dataloader(
        args.cifar,
        CIFAR_MEAN,
        CIFAR_STD,
        num_workers=args.w,
        batch_size=args.b,
        shuffle=args.s
    )


    model.cuda()
    model.eval()


    correct_1 = 0.0
    correct_5 = 0.0
    total = 0
    import time
    start=time.time()
    for n_iter, (image, label) in enumerate(cifar_test_loader):
        print("iteration: {}\ttotal {} iterations".format(n_iter + 1, len(cifar_test_loader)))
        image =image.cuda()
        label = label.cuda()
        output = model(image)
        _, pred = output.topk(5, 1, largest=True, sorted=True)

        label = label.view(label.size(0), -1).expand_as(pred)
        correct = pred.eq(label).float()

        #compute top 5
        correct_5 += correct[:, :5].sum()

        #compute top1
        correct_1 += correct[:, :1].sum()


    print()
    print("Top 1 acc: ", correct_1 / len(cifar_test_loader.dataset))
    print("Top 5 acc: ", correct_5 / len(cifar_test_loader.dataset))
    print(time.time() - start)
    print("Parameter numbers: {}".format(sum(p.numel() for p in model.parameters())))

    ##SAVE MODEL

    save_compress_model(model,'resnet.pth',cfg=cfg,mask=masks,codebook=codebooks)
    #print("model saved")